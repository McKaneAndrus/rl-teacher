import os.path as osp
from collections import deque

import numpy as np
import tensorflow as tf

CLIP_LENGTH = 1.5

def make_summary_writer(name):
    logs_path = osp.expanduser('~/tb/rl-teacher/%s' % (name))
    return tf.summary.FileWriter(logs_path)

def add_simple_summary(summary_writer, tag, simple_value, step):
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=simple_value)]), step)

class AgentLogger(object):
    """Tracks the performance of an arbitrary agent"""

    def __init__(self, summary_writer, timesteps_per_summary=int(1e3)):
        self.summary_step = 0
        self.timesteps_per_summary = timesteps_per_summary

        self._timesteps_elapsed = 0
        self._timesteps_since_last_training = 0

        n = 100
        self.last_n_paths = deque(maxlen=n)
        self.summary_writer = summary_writer

    def log_episode(self, path):
        self._timesteps_elapsed += len(path["obs"])
        self._timesteps_since_last_training += len(path["obs"])

        self.last_n_paths.append(path)

        if self._timesteps_since_last_training >= self.timesteps_per_summary:
            self.summary_step += 1
            last_n_episode_scores = [np.sum(path["original_rewards"]).astype(float) for path in self.last_n_paths]
            self.log_simple("agent/true_reward_per_episode", np.mean(last_n_episode_scores))
            self.log_simple("agent/total_steps", self._timesteps_elapsed, )

            self._timesteps_since_last_training -= self.timesteps_per_summary

    def log_simple(self, tag, simple_value, debug=False):
        add_simple_summary(self.summary_writer, tag, simple_value, self.summary_step)
        if debug:
            print("%s    =>    %s" % (tag, simple_value))