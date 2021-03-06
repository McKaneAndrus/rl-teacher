import os
import os.path as osp
import random
from collections import deque
from time import time, sleep
from ast import literal_eval

import numpy as np
import tensorflow as tf
from keras import backend as K
from parallel_trpo.train import train_parallel_trpo
from pposgd_mpi.run_mujoco import train_pposgd_mpi

from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.envs import get_timesteps_per_episode
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.label_schedules import LabelAnnealer, ConstantLabelSchedule
from rl_teacher.nn import FullyConnectedMLP
from rl_teacher.bnn import BNN
from rl_teacher.segment_sampling import sample_segment_from_path
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify, corrcoef, PiecewiseSchedule
from rl_teacher.video import SegmentVideoRecorder

from tensorflow.python import debug as tf_debug

CLIP_LENGTH = 1.5

class TraditionalRLRewardPredictor(object):
    """Predictor that always returns the true reward provided by the environment."""

    def __init__(self, summary_writer):
        self.agent_logger = AgentLogger(summary_writer)

    def predict_reward(self, path):
        self.agent_logger.log_episode(path)  # <-- This may cause problems in future versions of Teacher.
        return path["original_rewards"]

    def path_callback(self, path):
        pass

class ComparisonRewardPredictor():
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, env, summary_writer, comparison_collector, agent_logger, label_schedule, use_bnn,
                 bnn_samples, entropy_alpha, alpha_schedule, softmax_beta, beta_schedule,
                 trajectory_splits, info_gain_samples, random_sample_break, seed):
        self.summary_writer = summary_writer
        self.agent_logger = agent_logger
        self.comparison_collector = comparison_collector
        self.label_schedule = label_schedule
        self.use_bnn = use_bnn
        self.bnn_samples = bnn_samples
        self.use_entropy = entropy_alpha is not None or alpha_schedule is not None
        if not self.use_entropy:
            print("Not using entropy-seeking bonuses")
        else:
            print("Using entropy-seeking bonuses")
        self.entropy_alpha = alpha_schedule.value(0) if alpha_schedule is not None else entropy_alpha
        self.alpha_schedule = alpha_schedule
        self.softmax_beta = softmax_beta
        self.beta_schedule = beta_schedule
        self.trajectory_splits = trajectory_splits
        self.info_gain_samples = info_gain_samples
        self.random_sample_break = random_sample_break if random_sample_break is not None else float('inf')
        self.seed = seed
        random.seed(seed)


        # Set up some bookkeeping
        self.recent_segments = deque(maxlen=200)  # Keep a queue of recently seen segments to pull new comparisons from
        self._frames_per_segment = CLIP_LENGTH * env.fps
        self._steps_since_last_training = 0
        self._n_timesteps_per_predictor_training = 1e2  # How often should we train our predictor?
        self._elapsed_predictor_training_iters = 0

        # Build and initialize our predictor model
        tf.set_random_seed(seed)
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.InteractiveSession(config=config)
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.obs_shape = env.observation_space.shape
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape
        self.graph = self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _predict_rewards(self, obs_segments, act_segments, network):
        """
        :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
        :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
        :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
        :return: tensor with shape = (batch_size, segment_length)
        """
        batchsize = tf.shape(obs_segments)[0]
        segment_length = tf.shape(obs_segments)[1]

        # Temporarily chop up segments into individual observations and actions
        obs = tf.reshape(obs_segments, (-1,) + self.obs_shape)
        acts = tf.reshape(act_segments, (-1,) + self.act_shape)

        # Run them through our neural network
        rewards = network.run(obs, acts)

        # Group the rewards back into their segments
        return tf.reshape(rewards, (batchsize, segment_length))

    def _predict_bnn_rewards(self, obs_segments, act_segments, bayes_nn):
        batchsize = tf.shape(obs_segments)[0]
        segment_length = tf.shape(obs_segments)[1]

        obs = tf.reshape(obs_segments, (-1,) + self.obs_shape)
        acts = tf.reshape(act_segments, (-1,) + self.act_shape)

        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, acts], axis=1)


        # rewards = bayes_nn.construct_network(x)
        # rewards = tf.reshape(rewards, (batchsize, segment_length))

        # TODO make bayes_nn.run method
        network = bayes_nn.construct_network(x)
        network = tf.reshape(network, (batchsize, segment_length))
        rewards = tf.reduce_mean(bayes_nn.sample_network(network), axis=0)
        return rewards

    def _build_model(self):
        """
        Our model takes in path segments with states and actions, and generates Q values.
        These Q values serve as predictions of the true reward.
        We can compare two segments and sum the Q values to get a prediction of a label
        of which segment is better. We then learn the weights for our model by comparing
        these labels with an authority (either a human or synthetic labeler).
        """
        # Set up observation placeholders
        self.segment_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="obs_placeholder")
        self.segment_alt_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="alt_obs_placeholder")

        self.segment_act_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.act_shape, name="act_placeholder")
        self.segment_alt_act_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.act_shape, name="alt_act_placeholder")


        if self.use_bnn:
            print("Using BNN to generate more efficient queries")
            input_dim = np.prod(self.obs_shape) + np.prod(self.act_shape)
            self.rew_bnn = BNN(input_dim, [64, 64], 1, 10, self.sess, batch_size=1, trans_func=tf.nn.relu,
                               n_samples=self.bnn_samples, out_func=None)
            self.bnn_q_value = self._predict_bnn_rewards(self.segment_obs_placeholder, self.segment_act_placeholder, self.rew_bnn)
            bnn_alt_q_value = self._predict_bnn_rewards(self.segment_alt_obs_placeholder, self.segment_alt_act_placeholder, self.rew_bnn)
            # A vanilla multi-layer perceptron maps a (state, action) pair to a reward (Q-value)
        mlp = FullyConnectedMLP(self.obs_shape, self.act_shape)
        self.q_value = self._predict_rewards(self.segment_obs_placeholder, self.segment_act_placeholder, mlp)
        alt_q_value = self._predict_rewards(self.segment_alt_obs_placeholder, self.segment_alt_act_placeholder, mlp)

        print("Constructed Reward Model")

        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        segment_reward_pred_left = tf.reduce_sum(self.q_value, axis=1)
        segment_reward_pred_right = tf.reduce_sum(alt_q_value, axis=1)
        reward_logits = tf.stack([segment_reward_pred_left, segment_reward_pred_right], axis=1)  # (batch_size, 2)
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")

        # delta = 1e-5f
        # clipped_comparison_labels = tf.clip_by_value(self.comparison_labels, delta, 1.0-delta)

        self.data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=self.labels)

        self.loss_op = tf.reduce_mean(self.data_loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op, global_step=global_step)


        if self.use_bnn:
            segment_reward_bnn_left = tf.reduce_sum(self.bnn_q_value, axis=1)
            segment_reward_bnn_right = tf.reduce_sum(bnn_alt_q_value, axis=1)
            segment_reward_mean_left = tf.reduce_mean(self.bnn_q_value, axis=1)
            segment_reward_mean_right = tf.reduce_mean(bnn_alt_q_value, axis=1)
            # self.mean_rew_logits = tf.stack([segment_reward_mean_left, segment_reward_mean_right], axis=1)
            self.softmax_rew = tf.nn.softmax(reward_logits/self.softmax_beta)
            self.bnn_data_loss = self.rew_bnn.loss(segment_reward_bnn_left, segment_reward_bnn_right, self.labels)
            self.bnn_loss_op = tf.reduce_mean(self.bnn_data_loss)
            self.train_bnn_op = tf.train.AdamOptimizer().minimize(self.bnn_loss_op)
            self.plan_labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="plan_labels")
            self.planning_loss = self.rew_bnn.loss_last_sample(segment_reward_mean_left, segment_reward_mean_right,
                                                    self.plan_labels)
            self.planning_kl = self.rew_bnn.fast_kl_div(self.planning_loss, self.rew_bnn.get_mus(),
                                                    self.rew_bnn.get_rhos(), 0.01)

        print("Constructed Training Ops")

        return tf.get_default_graph()

    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            q_value = self.sess.run(self.q_value, feed_dict={
                self.segment_obs_placeholder: np.asarray([path["obs"]]),
                self.segment_act_placeholder: np.asarray([path["actions"]]),
                K.learning_phase(): False
            })
        return q_value[0]

    def compute_kl_term(self, path1, path2):
        return

    def path_callback(self, path):
        path_length = len(path["obs"])
        self._steps_since_last_training += path_length

        self.agent_logger.log_episode(path)

        # We may be in a new part of the environment, so we take new segments to build comparisons from
        segment = sample_segment_from_path(path, int(self._frames_per_segment))
        if segment:
            self.recent_segments.append(segment)

        # If we need more comparisons, then we build them from our recent segments
        if self.use_bnn and self._elapsed_predictor_training_iters < self.random_sample_break:
            self.rew_bnn.save_params()
            best_kl = float("-inf")
            best_a = 0
            best_b = 0
            pair_indices = np.random.randint(low=0, high=len(self.recent_segments), size=(self.info_gain_samples,2))
            for i in range(len(pair_indices)):
                a = pair_indices[i][0]
                b = pair_indices[i][1]
                if a == b:
                    continue
                else:
                    with self.graph.as_default():
                        seg1_obs = self.recent_segments[a]["obs"]
                        seg2_obs = self.recent_segments[b]["obs"]
                        seg1_acts = self.recent_segments[a]["actions"]
                        seg2_acts = self.recent_segments[b]["actions"]
                        kl1 = self.sess.run([self.planning_kl],
                            feed_dict={
                            self.segment_obs_placeholder: [seg1_obs],
                            self.segment_act_placeholder: [seg1_acts],
                            self.segment_alt_obs_placeholder: [seg2_obs],
                            self.segment_alt_act_placeholder: [seg2_acts],
                            self.plan_labels: [0],
                            K.learning_phase(): False
                            })
                        kl2 = self.sess.run([self.planning_kl],
                            feed_dict={
                            self.segment_obs_placeholder: [seg1_obs],
                            self.segment_act_placeholder: [seg1_acts],
                            self.segment_alt_obs_placeholder: [seg2_obs],
                            self.segment_alt_act_placeholder: [seg2_acts],
                            self.plan_labels: [1],
                            K.learning_phase(): False
                            })
                        prob = self.sess.run([self.softmax_rew],
                            feed_dict={
                            self.segment_obs_placeholder: [seg1_obs],
                            self.segment_act_placeholder: [seg1_acts],
                            self.segment_alt_obs_placeholder: [seg2_obs],
                            self.segment_alt_act_placeholder: [seg2_acts],
                            K.learning_phase(): False
                            })
                        p1 = prob[0][0][0]
                        p2 = prob[0][0][1]
                        kl1 = kl1[0]
                        kl2 = kl2[0]
                        #print("rewards ", p1, p2)
                        #print("kls ", kl1, kl2)
                        kl_val = p1*kl1 + p2*kl2
                        #print("KL: ", kl_val)
                        if kl_val > best_kl:
                            best_kl = kl_val
                            best_a = a
                            best_b = b
            # print("bestKL", best_kl, best_a, best_b)
            segments = [self.recent_segments[best_a], self.recent_segments[best_b]]
        else:
            segments = random.sample(self.recent_segments, 2) if len(self.recent_segments) > 2 else None


        if len(self.comparison_collector) < int(self.label_schedule.n_desired_labels):
            self.comparison_collector.add_segment_pair(segments[0], segments[1])

        # Train our predictor every X steps
        if self._steps_since_last_training >= int(self._n_timesteps_per_predictor_training):
            self.train_predictor()
            self._steps_since_last_training -= self._steps_since_last_training

    def train_predictor(self):
        self.comparison_collector.label_unlabeled_comparisons()

        minibatch_size = min(64, len(self.comparison_collector.labeled_decisive_comparisons))
        labeled_comparisons = random.sample(self.comparison_collector.labeled_decisive_comparisons, minibatch_size)
        left_obs = np.asarray([comp['left']['obs'] for comp in labeled_comparisons])
        left_acts = np.asarray([comp['left']['actions'] for comp in labeled_comparisons])
        right_obs = np.asarray([comp['right']['obs'] for comp in labeled_comparisons])
        right_acts = np.asarray([comp['right']['actions'] for comp in labeled_comparisons])
        labels = np.asarray([comp['label'] for comp in labeled_comparisons])

        with self.graph.as_default():
            _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict={
                self.segment_obs_placeholder: left_obs,
                self.segment_act_placeholder: left_acts,
                self.segment_alt_obs_placeholder: right_obs,
                self.segment_alt_act_placeholder: right_acts,
                self.labels: labels,
                K.learning_phase(): True
            })
            self._elapsed_predictor_training_iters += 1
            if self.alpha_schedule is not None:
                self.entropy_alpha = self.alpha_schedule.value(self._elapsed_predictor_training_iters)

            if self.beta_schedule is not None:
                self.softmax_beta = self.beta_schedule.value(self._elapsed_predictor_training_iters)

        if self.use_bnn:
            with self.graph.as_default():
                _, bnn_loss = self.sess.run([self.train_bnn_op, self.bnn_loss_op], feed_dict={
                    self.segment_obs_placeholder: left_obs,
                    self.segment_act_placeholder: left_acts,
                    self.segment_alt_obs_placeholder: right_obs,
                    self.segment_alt_act_placeholder: right_acts,
                    self.labels: labels,
                    K.learning_phase(): True
                })
                self._write_training_summaries(loss, bnn_loss)
        else:
            self._write_training_summaries(loss)




    def _write_training_summaries(self, loss, bnn_loss=None):
        self.agent_logger.log_simple("predictor/loss", loss)
        if bnn_loss is not None:
            self.agent_logger.log_simple("predictor/bnn_loss", bnn_loss)

        # Calculate correlation between true and predicted reward by running validation on recent episodes
        recent_paths = self.agent_logger.get_recent_paths_with_padding()
        if len(recent_paths) > 1 and self.agent_logger.summary_step % 10 == 0:  # Run validation every 10 iters
            validation_obs = np.asarray([path["obs"] for path in recent_paths])
            validation_acts = np.asarray([path["actions"] for path in recent_paths])
            q_value = self.sess.run(self.q_value, feed_dict={
                self.segment_obs_placeholder: validation_obs,
                self.segment_act_placeholder: validation_acts,
                K.learning_phase(): False
            })
            ep_reward_pred = np.sum(q_value, axis=1)
            reward_true = np.asarray([path['original_rewards'] for path in recent_paths])
            ep_reward_true = np.sum(reward_true, axis=1)
            self.agent_logger.log_simple("predictor/correlations", corrcoef(ep_reward_true, ep_reward_pred))

        self.agent_logger.log_simple("predictor/num_training_iters", self._elapsed_predictor_training_iters)
        self.agent_logger.log_simple("labels/desired_labels", self.label_schedule.n_desired_labels)
        self.agent_logger.log_simple("labels/total_comparisons", len(self.comparison_collector))
        self.agent_logger.log_simple(
            "labels/labeled_comparisons", len(self.comparison_collector.labeled_decisive_comparisons))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', required=True)
    parser.add_argument('-p', '--predictor', required=True)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-w', '--workers', default=6, type=int)
    parser.add_argument('-l', '--n_labels', default=None, type=int)
    parser.add_argument('-L', '--pretrain_labels', default=None, type=int)
    parser.add_argument('-t', '--num_timesteps', default=1e7, type=int)
    parser.add_argument('-a', '--agent', default="parallel_trpo", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=5000, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true")
    parser.add_argument('-b', '--use_bnn', action="store_true")
    parser.add_argument('-A', '--entropy_alpha', default=None, type=float)
    parser.add_argument('-As', '--alpha_schedule', default=None, type=str)
    parser.add_argument('-B', '--softmax_beta', default=1, type=float)
    parser.add_argument('-Bs', '--beta_schedule', default=None, type=str)
    parser.add_argument('-nb', '--num_bnn_samples', default=10, type=int)
    parser.add_argument('-ts', '--trajectory_splits', default=10, type=int)
    parser.add_argument('-ig', '--info_gain_samples', default=None, type=int)
    parser.add_argument('-rsb', '--random_sample_break',default=None,type=int)
    args = parser.parse_args()

    print("Setting things up...")

    env_id = args.env_id
    run_name = "%s/%s-%s" % (env_id, args.name, int(time()))
    summary_writer = make_summary_writer(run_name)

    env = make_with_torque_removed(env_id)

    num_timesteps = int(args.num_timesteps)
    experiment_name = slugify(args.name)


    if args.predictor == "rl":
        predictor = TraditionalRLRewardPredictor(summary_writer)
    else:
        agent_logger = AgentLogger(summary_writer)

        pretrain_labels = args.pretrain_labels if args.pretrain_labels else args.n_labels // 4

        if args.n_labels:
            label_schedule = LabelAnnealer(
                agent_logger,
                final_timesteps=num_timesteps,
                final_labels=args.n_labels,
                pretrain_labels=pretrain_labels)
        else:
            print("No label limit given. We will request one label every few seconds.")
            label_schedule = ConstantLabelSchedule(pretrain_labels=pretrain_labels)

        if args.predictor == "synth":
            comparison_collector = SyntheticComparisonCollector()

        elif args.predictor == "human":
            bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
            assert bucket and bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"
            comparison_collector = HumanComparisonCollector(env_id, experiment_name=experiment_name)
        else:
            raise ValueError("Bad value for --predictor: %s" % args.predictor)

        if args.alpha_schedule is not None:
            print(literal_eval(args.alpha_schedule))
            temp = literal_eval(args.alpha_schedule)
            temp = [(0,temp[0][1])] + [(tup[0] + args.pretrain_iters,tup[1]) for tup in temp]
            print(temp)
            alpha_schedule = PiecewiseSchedule(temp, outside_value=temp[-1][1])
        else:
            alpha_schedule = None

        if args.beta_schedule is not None:
            print(literal_eval(args.beta_schedule))
            temp = literal_eval(args.beta_schedule)
            temp = [(0,temp[0][1])] + [(tup[0] + args.pretrain_iters, tup[1]) for tup in temp]
            print(temp)
            beta_schedule = PiecewiseSchedule(temp, outside_value=temp[-1][1])
        else:
            beta_schedule = None
        predictor = ComparisonRewardPredictor(
            env,
            summary_writer,
            comparison_collector=comparison_collector,
            agent_logger=agent_logger,
            label_schedule=label_schedule,
            use_bnn=args.use_bnn,
            bnn_samples = args.num_bnn_samples,
            entropy_alpha = args.entropy_alpha,
            alpha_schedule=alpha_schedule,
            softmax_beta = args.softmax_beta,
            beta_schedule = beta_schedule,
            trajectory_splits = args.trajectory_splits,
            info_gain_samples=args.info_gain_samples,
            random_sample_break = args.random_sample_break,
            seed=args.seed
        )

        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        pretrain_segments = segments_from_rand_rollout(
            env_id, make_with_torque_removed, n_desired_segments=pretrain_labels * 2,
            clip_length_in_seconds=CLIP_LENGTH, workers=args.workers)
        for i in range(pretrain_labels):  # Turn our random segments into comparisons
            comparison_collector.add_segment_pair(pretrain_segments[i], pretrain_segments[i + pretrain_labels])

        # Sleep until the human has labeled most of the pretraining comparisons
        while len(comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):
            comparison_collector.label_unlabeled_comparisons()
            if args.predictor == "synth":
                print("%s synthetic labels generated... " % (len(comparison_collector.labeled_comparisons)))
            elif args.predictor == "human":
                print("%s/%s comparisons labeled. Please add labels w/ the human-feedback-api. Sleeping... " % (
                    len(comparison_collector.labeled_comparisons), pretrain_labels))
                sleep(5)

        # Start the actual training
        for i in range(args.pretrain_iters):
            predictor.train_predictor()  # Train on pretraining labels
            if i % 100 == 0:
                print("%s/%s predictor pretraining iters... " % (i, args.pretrain_iters))

    # Wrap the predictor to capture videos every so often:
    if not args.no_videos:
        predictor = SegmentVideoRecorder(predictor, env, save_dir=osp.join('/tmp/rl_teacher_vids', run_name))

    # We use a vanilla agent from openai/baselines that contains a single change that blinds it to the true reward
    # The single changed section is in `rl_teacher/agent/trpo/core.py`
    print("Starting joint training of predictor and agent")
    if args.agent == "parallel_trpo":
        train_parallel_trpo(
            env_id=env_id,
            make_env=make_with_torque_removed,
            predictor=predictor,
            summary_writer=summary_writer,
            workers=args.workers,
            runtime=-1,
            max_timesteps_per_episode=get_timesteps_per_episode(env),
            timesteps_per_batch=8000,
            max_kl=0.001,
            seed=args.seed,
        )
    elif args.agent == "pposgd_mpi":
        def make_env():
            return make_with_torque_removed(env_id)

        train_pposgd_mpi(make_env, num_timesteps=num_timesteps, seed=args.seed, predictor=predictor)
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)

if __name__ == '__main__':
    main()

