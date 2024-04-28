# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class RBM(object):
    def __init__(self, n_visibe, n_hidden, layer_names, learning_rate=0.01):
        """
        :param n_visibe: number of visible units
        :param n_hidden: number of hidden units
        :param layer_names: layers names
        :param learning_rate:optional, default = 0.01
        """
        # Initialize params
        self.n_visible = n_visibe
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.layer_names = layer_names

        self.weights_and_biases = self._initialize_weights_and_biases()
        self.visible_0, self.rbm_W, self.rbm_a, self.rbm_b = self._create_placeholders()

        # Weights initialization

        self.n_w = np.zeros([self.n_visible, self.n_hidden], np.float32)
        self.n_vb = np.zeros([self.n_visible], np.float32)
        self.n_hb = np.zeros([self.n_hidden], np.float32)
        self.o_w = np.random.normal(0.0, 0.01, [self.n_visible, self.n_hidden])
        self.o_vb = np.zeros([self.n_visible], np.float32)
        self.o_hb = np.zeros([self.n_hidden], np.float32)

        self.hprobs_0, self.hstates_0, self.visible_1, self.hprobs_1 = (
            self._gibbs_sampling_step()
        )
        self.positive_grad, self.negative_grad = self._compute_gradients()
        self.update_W, self.update_a, self.update_b = self._update_weights_and_biases()

        # sampling functions
        self.h_sample = tf.nn.sigmoid(
            tf.matmul(self.visible_0, self.rbm_W) + self.rbm_b
        )
        self.v_sample = tf.nn.sigmoid(
            tf.matmul(self.h_sample, tf.transpose(self.rbm_W)) + self.rbm_a
        )

        self.error = self._compute_cost()

        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _initialize_weights_and_biases(self):
        weights_and_biases = {
            # Weights are initialized to small random values chosen from a zero-mean Gaussian with a standard deviation of 0.01
            # Large random values increases the speed of the training but results in slightly worse final model.
            "W": tf.Variable(
                tf.random_normal(
                    [self.n_visible, self.n_hidden], stddev=0.01, dtype=tf.float32
                ),
                name=self.layer_names[0],
            ),
            # a - bias of visible units; b - bias of hidden units
            # Hidden biases are initialized to zero. Helpful to initialize the bias of visible unit i to log[pi/(1−pi)]
            "a": tf.Variable(
                tf.zeros([self.n_visible], dtype=tf.float32), name=self.layer_names[1]
            ),
            "b": tf.Variable(
                tf.zeros([self.n_hidden], dtype=tf.float32), name=self.layer_names[2]
            ),
        }

        return weights_and_biases

    def _create_placeholders(self):
        visible_0 = tf.placeholder(tf.float32, [None, self.n_visible])
        rbm_W = tf.placeholder(tf.float32, [self.n_visible, self.n_hidden])
        # Hidden bias and visible bias
        rbm_a = tf.placeholder(tf.float32, [self.n_visible])
        rbm_b = tf.placeholder(tf.float32, [self.n_hidden])
        return visible_0, rbm_W, rbm_a, rbm_b

    def _gibbs_sampling_step(self):
        hprobs_0 = tf.nn.sigmoid(tf.matmul(self.visible_0, self.rbm_W) + self.rbm_b)
        hstates_0 = self._sample_probs(hprobs_0)

        # It is common to use the probability, pi, instead of sampling a binary value
        visible_1 = tf.nn.sigmoid(
            tf.matmul(hprobs_0, tf.transpose(self.rbm_W)) + self.rbm_a
        )
        # When hidden units are being driven by reconstructions, always use probabilities without sampling.
        hprobs_1 = tf.nn.sigmoid(tf.matmul(visible_1, self.rbm_W) + self.rbm_b)
        # hstates_1 = self._sample_probs(hprobs_1)

        return hprobs_0, hstates_0, visible_1, hprobs_1

    def _sample_probs(self, probs):
        """

        :param probs: tensor of probabilities
        :return: binary states
        """

        # The hidden unit turns on if this probability is greater than a random number uniformly distributed between 0 and 1
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def _compute_gradients(self):
        positive_grad = tf.matmul(tf.transpose(self.visible_0), self.hstates_0)
        negative_grad = tf.matmul(tf.transpose(self.visible_1), self.hprobs_1)

        return positive_grad, negative_grad

    def _update_weights_and_biases(self):
        update_W = self.rbm_W + self.learning_rate * (
            self.positive_grad - self.negative_grad
        ) / tf.to_float(tf.shape(self.visible_0)[0])
        update_a = self.rbm_a + self.learning_rate * tf.reduce_mean(
            self.visible_0 - self.visible_1, 0
        )
        update_b = self.rbm_b + self.learning_rate * tf.reduce_mean(
            self.hprobs_0 - self.hprobs_1, 0
        )
        return update_W, update_a, update_b

    def restore_weights(self, path):
        saver = tf.train.Saver(
            {
                self.layer_names[0]: self.weights_and_biases["W"],
                self.layer_names[1]: self.weights_and_biases["a"],
                self.layer_names[2]: self.weights_and_biases["b"],
            }
        )

        saver.restore(self.sess, path)

        self.o_w = self.weights_and_biases["W"].eval(self.sess)
        self.o_vb = self.weights_and_biases["a"].eval(self.sess)
        self.o_hb = self.weights_and_biases["b"].eval(self.sess)

    def save_weights(self, path):
        self.sess.run(self.weights_and_biases["W"].assign(self.o_w))
        self.sess.run(self.weights_and_biases["a"].assign(self.o_vb))
        self.sess.run(self.weights_and_biases["b"].assign(self.o_hb))

        saver = tf.train.Saver(
            {
                self.layer_names[0]: self.weights_and_biases["W"],
                self.layer_names[1]: self.weights_and_biases["a"],
                self.layer_names[2]: self.weights_and_biases["b"],
            }
        )
        saver.save(self.sess, path)

    def _compute_cost(self):
        return tf.reduce_mean(tf.square(self.visible_0 - self.v_sample))

    def compute_cost(self, batch):
        return self.sess.run(
            self.error,
            feed_dict={
                self.visible_0: batch,
                self.rbm_W: self.o_w,
                self.rbm_a: self.o_vb,
                self.rbm_b: self.o_hb,
            },
        )

    def partial_fit(self, batch_x):
        #  It is often more efficient to divide the training set into small “mini-batches” of 10 to 100 cases
        self.n_w, self.n_vb, self.n_hb = self.sess.run(
            [self.update_W, self.update_a, self.update_b],
            feed_dict={
                self.visible_0: batch_x,
                self.rbm_W: self.o_w,
                self.rbm_a: self.o_vb,
                self.rbm_b: self.o_hb,
            },
        )

        self.o_w = self.n_w
        self.o_vb = self.n_vb
        self.o_hb = self.n_hb

        return self.sess.run(
            self.error,
            feed_dict={
                self.visible_0: batch_x,
                self.rbm_W: self.n_w,
                self.rbm_a: self.n_vb,
                self.rbm_b: self.n_hb,
            },
        )

    def transform(self, batch_x):
        return self.sess.run(
            self.h_sample,
            {
                self.visible_0: batch_x,
                self.rbm_W: self.o_w,
                self.rbm_a: self.o_vb,
                self.rbm_b: self.o_hb,
            },
        )
