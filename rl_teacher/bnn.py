from __future__ import print_function
import numpy as np
import tensorflow as tf
import keras
from collections import OrderedDict
from keras.engine.topology import Layer
from keras import backend as K

# --------------------
BNN_LAYER_TAG = 'BNNLayer'
USE_REPARAMETERIZATION_TRICK = True
# --------------------


class BNNLayer():
	def __init__(self, inputs, outputs, session, scope, nonlinearity=tf.nn.relu, prior_sd=None):
		self.nonlin = nonlinearity
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
	        self.mu = tf.get_variable("mu", [inputs, outputs], initializer=tf.random_normal_initializer(), trainable=True)
	        self.rho = tf.get_variable("rho", [inputs, outputs], initializer=tf.constant_initializer(prior_rho), trainable=True)

	        # Bias Priors
	        self.b_mu = tf.get_variable("b_mu", [outputs], initializer=tf.random_normal_initializer(), trainable=True)
	        self.b_rho = tf.get_variable("b_rho", [outputs], initializer=tf.constant_initializer(prior_rho), trainable=True)

	        # Backup Params for KL Calculation

	        self.mu_old = tf.get_variable("mu_old", [inputs, outputs], initializer=tf.zeros_initializer)
	        self.rho_old = tf.get_variable("rho_old", [inputs, outputs], initializer=tf.ones_initializer)

	        self.b_mu_old = tf.get_variable("b_mu_old", [outputs], initializer=tf.zeros_initializer)
	        self.b_rho_old = tf.get_variable("b_rho_old", [outputs], initializer=tf.ones_initializer)
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
		epsilon = np.random.normal(size=self.shape)
		W = self.mu + self.log_to_std(self.rho) * epsilon
		self.W = W
		return W

	def get_b_const(self):
		epsilon = np.random.normal(size=self.num_outputs)
		b = self.b_mu_ary + self.log_to_std_np(self.b_rho_ary) * epsilon
		self.b_const = b
		return b

	def get_b(self):
		epsilon = np.random.normal(size=self.num_outputs)
		b = self.b_mu + self.log_to_std(self.b_rho) * epsilon
		self.b = b
		return b

	def save_old_params(self):
		op1 = self.mu_old.assign(self.mu_ary)
		op2 = self.rho_old.assign(self.rho_ary)
		op3 = self.b_mu_old.assign(self.b_mu_ary)
		op4 = self.b_rho_old.assign(self.b_rho_ary)
		self.sess.run([op1, op2, op3, op4])

	def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
		"""KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        numerator = tf.square(p_mean - q_mean) + \
            tf.square(p_std) - tf.square(q_std)
        denominator = 2 * tf.square(q_std) + 1e-8
        return tf.reduce_sum(
            numerator / denominator + tf.log(q_std) - tf.log(p_std))

    def kl_div_new_old(self):
        kl_div = self.kl_div_p_q(
            self.mu, self.log_to_std(self.rho), self.mu_old, self.log_to_std(self.rho_old))
        kl_div += self.kl_div_p_q(self.b_mu, self.log_to_std(self.b_rho),
                                  self.b_mu_old, self.log_to_std(self.b_rho_old))
        return kl_div

    def kl_div_old_new(self):
        kl_div = self.kl_div_p_q(
            self.mu_old, self.log_to_std(self.rho_old), self.mu, self.log_to_std(self.rho))
        kl_div += self.kl_div_p_q(self.b_mu_old,
                                  self.log_to_std(self.b_rho_old), self.b_mu, self.log_to_std(self.b_rho))
        return kl_div

    def kl_div_new_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu, self.log_to_std(self.rho), 0., self.prior_sd)
        kl_div += self.kl_div_p_q(self.b_mu,
                                  self.log_to_std(self.b_rho), 0., self.prior_sd)
        return kl_div

    def kl_div_old_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu_old, self.log_to_std(self.rho_old), 0., self.prior_sd)
        kl_div += self.kl_div_p_q(self.b_mu_old,
                                  self.log_to_std(self.b_rho_old), 0., self.prior_sd)
        return kl_div

    def kl_div_prior_new(self):
        kl_div = self.kl_div_p_q(
            0., self.prior_sd, self.mu,  self.log_to_std(self.rho))
        kl_div += self.kl_div_p_q(0., self.prior_sd,
                                  self.b_mu, self.log_to_std(self.b_rho))
        return kl_div

class BNN():

	 def __init__(self, n_in,
                 n_hidden,
                 n_out,
                 layers_type,
                 n_batches,
                 trans_func=tf.nn.relu,
                 out_func=None,
                 batch_size=100,
                 n_samples=10,
                 prior_sd=0.5,
                 likelihood_sd=5.0,
                 second_order_update=True,
                 learning_rate=0.0001,
                 information_gain=True,
                 ):

        Serializable.quick_init(self, locals())
        assert len(layers_type) == len(n_hidden) + 1

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.batch_size = batch_size
        self.transf = trans_func
        self.outf = out_func
        self.n_samples = n_samples
        self.prior_sd = prior_sd
        self.layers_type = layers_type
        self.n_batches = n_batches
        self.likelihood_sd = likelihood_sd
        self.second_order_update = second_order_update
        self.learning_rate = learning_rate
        self.information_gain = information_gain

        assert self.information_gain

        # Build network architecture.
        self.build_network()

	def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        return tf.reduce_sum(l.kl_div_new_prior() for l in self.layers)

	def log_prob_label(prediction1, prediction2, target):
		if target == 0:
			return tf.log(tf.exp(prediction1)/(tf.exp(prediction2) + tf.exp(prediction1)))
		else:
			return tf.log(tf.exp(prediction2)/(tf.exp(prediction2) + tf.exp(prediction1)))

	def pred_sym(input_ph):
		return construct_network(input_ph)


	def loss(self, input1_ph, input2_ph, target):

        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction1 = self.pred_sym(input1_ph)
            prediction2 = self.pred_sym(input2_ph)
            # Calculate model likelihood log(P(D|w)).
            _log_p_D_given_w.append(self.log_prob_label(
                prediction1, prediction2, target))
        log_p_D_given_w = tf.reduce_sum(_log_p_D_given_w)
        # Calculate variational posterior log(q(w)) and prior log(p(w)).
        kl = self.log_p_w_q_w_kl()
        # if self.use_reverse_kl_reg:
        #     kl += self.reverse_kl_reg_factor * \
        #         self.reverse_log_p_w_q_w_kl()

        # Calculate loss function.
        return kl / self.n_batches - log_p_D_given_w / self.n_samples

    def loss_last_sample(self, input1, input2, target):
        """The difference with the original loss is that we only update based on the latest sample.
        This means that instead of using the prior p(w), we use the previous approximated posterior
        q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)].
        """

        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
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
        for i in xrange(len(self.n_hidden)):
            # Probabilistic layer (1) or deterministic layer (0).
            name = "layer%d"%i
            if i==0:
            	l = BNNLayer(self.n_in, self.n_hidden[i], self.sess, name, nonlinearity=self.transf, prior_sd=self.prior_sd, name=BNN_LAYER_TAG)
            	self.layers.append(l)
           	else:
            	l = BNNLayer(self.n_hidden[i-1], self.n_hidden[i], self.sess, name, nonlinearity=self.transf, prior_sd=self.prior_sd, name=BNN_LAYER_TAG)
            	self.layers.append(l)

        # Output layer
        name = "layer%s"%"out"
        l_out = BNNLayer(self.n_hidden[-1], self.n_out, self.sess, name, nonlinearity=self.outf, prior_sd=self.prior_sd, name=BNN_LAYER_TAG)
        self.layers.append(l_out)
        self.Ws = []
        self.bs = []
        for layer in self.layers:
        	self.Ws.append(layer.get_W())
        	self.bs.append(layer.get_b())

    def construct_network(self, input_ph):
    	network = input_ph

    	for i in xrange(len(self.layers)):
            # Probabilistic layer (1) or deterministic layer (0).
            l_W = self.Ws[i]
            l_b = self.bs[i]
            network = tf.matmul(network, l_W) + l_b

        return network

    def fast_kl_div(loss, params, oldparams, step_size):

        grads = tf.gradients(loss, params)

        kl_component = []
        for i in xrange(len(params)):
            param = params[i]
            grad = grads[i]

            if param.name == 'mu' or param.name == 'b_mu':
                oldparam_rho = oldparams[i + 1]
                invH = tf.square(tf.log(1 + tf.exp(oldparam_rho)))
            else:
                oldparam_rho = oldparams[i]
                p = param
                H = 2. * (tf.exp(2 * p)) / (1 + tf.exp(p))**2 / (tf.log(1 + tf.exp(p))**2)
                invH = 1. / H
            kl_component.append(
                tf.add(tf.square(step_size) * tf.square(grad) * invH))

        return tf.reduce_sum(kl_component)
