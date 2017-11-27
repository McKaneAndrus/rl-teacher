from __future__ import print_function
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from keras.engine.topology import Layer
from keras import backend as K

# --------------------
BNN_LAYER_TAG = 'BNNLayer'
USE_REPARAMETERIZATION_TRICK = True
# --------------------


class BNNLayer():
	def __init__(self, inputs, outputs, session, scope, nonlinearity=tf.nn.relu, prior_sd=None):
		self.non_lin = nonlinearity
		self.num_inputs = inputs
		self.num_outputs = outputs
		self.sess = session
		prior_rho = self.std_to_log_np(prior_sd)
		self.prior_rho = prior_rho
		self.prior_sd = prior_sd
		self.shape = [inputs, outputs]
		self.scope=scope

		with tf.variable_scope(self.scope):
			self.W = None
			self.b = None
	        # Priors
			self.mu = tf.get_variable("mu", [inputs, outputs], initializer=tf.random_normal_initializer(), dtype=tf.float32, trainable=True)
			self.rho = tf.get_variable("rho", [inputs, outputs], initializer=tf.constant_initializer(prior_rho), dtype=tf.float32, trainable=True)

	        # Bias Priors
			self.b_mu = tf.get_variable("b_mu", [outputs], initializer=tf.random_normal_initializer(), dtype=tf.float32, trainable=True)
			self.b_rho = tf.get_variable("b_rho", [outputs], initializer=tf.constant_initializer(prior_rho), dtype=tf.float32, trainable=True)

	        # Backup Params for KL Calculation

	        # self.mu_old = tf.get_variable("mu_old", [inputs, outputs], initializer=tf.zeros_initializer)
	        # self.rho_old = tf.get_variable("rho_old", [inputs, outputs], initializer=tf.ones_initializer)

	        # self.b_mu_old = tf.get_variable("b_mu_old", [outputs], initializer=tf.zeros_initializer)
	        # self.b_rho_old = tf.get_variable("b_rho_old", [outputs], initializer=tf.ones_initializer)

			self.mu_old = np.zeros(shape=(inputs, outputs), dtype=np.float32)
			self.rho_old = np.ones(shape=(inputs, outputs), dtype=np.float32)
			self.b_mu_old = np.zeros(shape=(outputs), dtype=np.float32)
			self.b_rho_old = np.ones(shape=(outputs), dtype=np.float32)
			self.sess.run(tf.global_variables_initializer())
			self.mu_ary, self.rho_ary, self.b_mu_ary, self.b_rho_ary = self.sess.run([self.mu, self.rho, self.b_mu, self.b_rho])

	def log_to_std_np(self, rho):
		"""Transformation to keep std non negative"""
		return np.log(1 + np.exp(rho))

	def std_to_log_np(self, sigma):
		"""Transform back into rho"""
		return np.log(np.exp(sigma)-1)

	def log_to_std(self, rho):
		"""Transformation to keep std non negative"""
		return tf.log(1 + tf.exp(rho))

	def std_to_log(self, sigma):
		"""Transform back into rho"""
		return tf.log(tf.exp(sigma)-1)

	def get_W_const(self):
		epsilon = np.random.normal(size=self.shape)
		W = self.mu_ary + self.log_to_std_np(self.rho_ary) * epsilon
		self.W_const = W
		return W

	def get_W(self):
		epsilon = tf.random_normal(shape=self.shape)
		#print("EPSILON", epsilon)
		W = self.mu + self.log_to_std(self.rho) * epsilon
		#W = self.mu
		self.W = W
		return W

	def get_b_const(self):
		epsilon = np.random.normal(size=self.num_outputs)
		b = self.b_mu_ary + self.log_to_std_np(self.b_rho_ary) * epsilon
		self.b_const = b
		return b

	def get_b(self):
		epsilon = tf.random_normal(shape=[self.num_outputs])
		b = self.b_mu + self.log_to_std(self.b_rho) * epsilon
		#b = self.b_mu
		self.b = b
		return b

	# def save_old_params(self):
	# 	op1 = self.mu_old.assign(self.mu_ary)
	# 	op2 = self.rho_old.assign(self.rho_ary)
	# 	op3 = self.b_mu_old.assign(self.b_mu_ary)
	# 	op4 = self.b_rho_old.assign(self.b_rho_ary)
	# 	self.sess.run([op1, op2, op3, op4])

	def save_old_params(self):
		self.mu_old = self.mu_ary
		self.rho_old = self.rho_ary
		self.b_mu_old = self.b_mu_ary
		self.b_rho_old = self.b_rho_ary

	def reset_to_old_params(self):
		op1 = self.mu.assign(self.mu_old)
		op2 = self.rho.assign(self.rho_old)
		op3 = self.b_mu.assign(self.b_mu_old)
		op4 = self.b_rho.assign(self.b_rho_old)
		self.sess.run([op1, op2, op3, op4])

	def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
		"""KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
		numerator = tf.square(p_mean - q_mean) + tf.square(p_std) - tf.square(q_std)
		denominator = 2 * tf.square(q_std) + 1e-8
		return tf.reduce_sum(numerator / denominator + tf.log(q_std) - tf.log(p_std))

	def kl_div_new_old(self):
		kl_div = self.kl_div_p_q(
			self.mu, self.log_to_std(self.rho), self.mu_old, self.log_to_std(self.rho_old))
		kl_div += self.kl_div_p_q(self.b_mu, self.log_to_std(self.b_rho),
			self.b_mu_old, self.log_to_std(self.b_rho_old))
		return kl_div

	def kl_div_old_new(self):
		kl_div = self.kl_div_p_q(
			self.mu_old, self.log_to_std(self.rho_old), self.mu, self.log_to_std(self.rho))
		kl_div += self.kl_div_p_q(
			self.b_mu_old,self.log_to_std(self.b_rho_old), self.b_mu, self.log_to_std(self.b_rho))
		return kl_div

	def kl_div_new_prior(self):
		kl_div = self.kl_div_p_q(self.mu, self.log_to_std(self.rho), 0., self.prior_sd)
		kl_div += self.kl_div_p_q(self.b_mu, self.log_to_std(self.b_rho), 0., self.prior_sd)
		return kl_div

	def kl_div_old_prior(self):
		kl_div = self.kl_div_p_q(self.mu_old, self.log_to_std(self.rho_old), 0., self.prior_sd)
		kl_div += self.kl_div_p_q(self.b_mu_old, self.log_to_std(self.b_rho_old), 0., self.prior_sd)
		return kl_div

	def kl_div_prior_new(self):
		kl_div = self.kl_div_p_q(0., self.prior_sd, self.mu,  self.log_to_std(self.rho))
		kl_div += self.kl_div_p_q(0., self.prior_sd, self.b_mu, self.log_to_std(self.b_rho))
		return kl_div

class BNN():

	def __init__(self, n_in,
                 n_hidden,
                 n_out,
                 n_batches,
                 session,
                 trans_func=tf.nn.relu,
                 out_func=None,
                 batch_size=100,
                 n_samples=10,
                 prior_sd=0.5,
                 likelihood_sd=5.0,
                 second_order_update=True,
                 learning_rate=0.0001,
                 information_gain=True):

		self.n_in = n_in
		self.n_hidden = n_hidden
		self.n_out = n_out
		self.batch_size = batch_size
		self.transf = trans_func
		self.outf = out_func
		self.n_samples = n_samples
		self.prior_sd = prior_sd
		self.n_batches = n_batches
		self.likelihood_sd = likelihood_sd
		self.second_order_update = second_order_update
		self.learning_rate = learning_rate
		self.information_gain = information_gain
		self.sess = session
		assert self.information_gain

        # Build network architecture.
		self.build_network()

	def get_rhos(self):
		rhos = []
		for l in self.layers:
			rhos.append(l.rho)
		self.rhos = np.array(rhos)
		return self.rhos

	def get_mus(self):
		mus = []
		for l in self.layers:
			mus.append(l.mu)
		self.mus = np.array(mus)
		return self.mus

	def get_old_rhos(self):
		rhos = []
		for l in self.layers:
			rhos.append(l.rho_old)
		self.old_rhos = np.array(rhos)
		return self.old_rhos

	def get_old_mus(self):
		mus = []
		for l in self.layers:
			mus.append(l.mu_old)
		self.old_mus = np.array(mus)
		return self.old_mus

	def log_p_w_q_w_kl(self):
		"""KL divergence KL[q_\phi(w)||p(w)]"""
		return tf.reduce_sum([l.kl_div_new_prior() for l in self.layers])

	def log_prob_label(self, prediction1, prediction2, target):
		print("PREDIUCTION", tf.shape(prediction1))
		reward_logits = tf.stack([prediction1, prediction2], axis=1)
		print("LOGIT SHAPE", tf.shape(reward_logits))
		return tf.log(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=target)+1e-8)

	def _log_prob_normal(self, input, mu=0., sigma=1.):
		log_normal = -tf.log(sigma) - tf.log(tf.sqrt(2 * np.pi)) - tf.square(input - mu) / (2 * tf.square(sigma))
		return tf.reduce_sum(log_normal)

	def pred_sym(self, input_ph, batchsize, segment_length):
		rewards = self.construct_network(input_ph)
		tf.reshape(rewards, (batchsize, segment_length))
		return tf.reduce_sum(rewards, axis=1)

	def refresh_weights(self):
		self.Ws = []
		self.bs = []
		for layer in self.layers:
			self.Ws.append(layer.get_W())
			self.bs.append(layer.get_b())

	def loss(self, input1_ph, input2_ph, target):
		# MC samples.
		_log_p_D_given_w = []
		for _ in range(self.n_samples):
			# Make prediction.
			prediction1 = input1_ph
			prediction2 = input2_ph
			# Calculate model likelihood log(P(D|w)).
			prob = self.log_prob_label(prediction1, prediction2, target)
			print("PROB", prob)
			_log_p_D_given_w.append(prob)
			#self.refresh_weights()
		log_p_D_given_w = tf.reduce_sum(_log_p_D_given_w)
		# Calculate variational posterior log(q(w)) and prior log(p(w)).
		kl = self.log_p_w_q_w_kl()
        # if self.use_reverse_kl_reg:
        #     kl += self.reverse_kl_reg_factor * \
        #         self.reverse_log_p_w_q_w_kl()

        # Calculate loss function.
		print("LOSS", kl, log_p_D_given_w)
		return kl - log_p_D_given_w / self.n_samples

	def loss_last_sample(self, input1, input2, target):
		"""The difference with the original loss is that we only update based on the latest sample.
		This means that instead of using the prior p(w), we use the previous approximated posterior
		q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)].
		"""

        # MC samples.
		_log_p_D_given_w = []
		for _ in range(self.n_samples):
		    # Make prediction.
		    prediction1 = self.pred_sym(input1)
		    prediction2 = self.pred_sym(input2)
		    # Calculate model likelihood log(P(sample|w)).
		    _log_p_D_given_w.append(self.log_prob_label(
		        prediction1, prediction2, target))
		log_p_D_given_w = tf.reduce_sum(_log_p_D_given_w)
		# Calculate loss function.
		# self.kl_div() should be zero when taking second order step
		return - log_p_D_given_w / self.n_samples

	def build_network(self):

		# Input layer
		self.layers = []
		# Hidden layers
		for i in range(len(self.n_hidden)):
			# Probabilistic layer (1) or deterministic layer (0).
			name = "layer%d"%i
			if i==0:
				l = BNNLayer(self.n_in, self.n_hidden[i], self.sess, scope=name, nonlinearity=self.transf, prior_sd=self.prior_sd)
				self.layers.append(l)
			else:
				l = BNNLayer(self.n_hidden[i-1], self.n_hidden[i], self.sess, scope=name, nonlinearity=self.transf, prior_sd=self.prior_sd)
				self.layers.append(l)

		# Output layer
		name = "layer%s"%"out"
		l_out = BNNLayer(self.n_hidden[-1], self.n_out, self.sess, scope=name, nonlinearity=self.outf, prior_sd=self.prior_sd)
		self.layers.append(l_out)
		self.Ws = []
		self.bs = []
		for layer in self.layers:
			self.Ws.append(layer.get_W())
			self.bs.append(layer.get_b())

	def construct_network(self, input_ph):
		network = input_ph
		#print(self.layers)
		for i in range(len(self.layers)):
			# Probabilistic layer (1) or deterministic layer (0).
			l = self.layers[i]
			l_W = self.Ws[i]
			l_b = self.bs[i]
			if l.non_lin:
				network = l.non_lin(tf.matmul(network, l_W) + l_b)
			else:
				network = tf.matmul(network, l_W) + l_b

		return network

	def fast_kl_div(self, loss, mus, rhos, mu_olds, rho_olds, step_size):

		grads_mu = [tf.gradients(loss, mu) for mu in mus]
		grads_rho = [tf.gradients(loss, rho) for rho in rhos]

		kl_components = []
		for i in range(len(mus)):
			grad = grads_mu[i]
			old_rho = rho_olds[i]
			invH = tf.square(tf.log(1.0 + tf.exp(old_rho)))
			kl_components.append(tf.reduce_sum(tf.square(step_size) * tf.square(grad) * invH))
		for j in range(len(rhos)):
			grad = grads_rho[j]
			rho = rhos[j]
			H = 2.0 * (tf.exp(2.0 * rho)) / (1.0 + tf.exp(rho))**2.0 / (tf.log(1 + tf.exp(rho))**2)
			invH = 1.0 / H
			kl_components.append(tf.reduce_sum(tf.square(step_size) * tf.square(grad) * invH))
		return tf.reduce_sum(kl_components)
