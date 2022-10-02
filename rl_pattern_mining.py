from __future__ import division
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


class Actor(object):
    
    ### THIS ACTOR ONLY WORK FOR PATTERN MINING!!!
    ### THE OUT OF ACTOR IS AVERAGED OVER AXIS 0 (BATCH SIZE)
    
    def __init__(self, graph, state_dim, action_dim, learning_rate, tau, batch_size, save_path):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        self.tau = tau
        self.batch_size = batch_size
        self.save_path = save_path
        self.is_training = tf.placeholder(shape=[], dtype=tf.bool)
        with graph.as_default():

            # Actor
            self.inputs, self.out = self.create_actor("Actor")
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/")

            # Target
            self.target_inputs, self.target_out = self.create_actor("Actor_target", False)
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Actor_target/")
            self.update_target_network_params = [self.target_network_params[i]
                                                 .assign(tf.multiply(self.network_params[i], self.tau) 
                                                         + tf.multiply(self.target_network_params[i], 1. - self.tau))
                                                 for i in range(len(self.target_network_params))]


            # Action Gradient
            self.action_gradient = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)

            # Policy Gradient
            self.unnormalized_actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradient)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

            # Train
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "Actor/")
            self.optimize = tf.train.AdamOptimizer(self.learning_rate)
            with tf.control_dependencies(self.update_ops):
                self.optimize = self.optimize.apply_gradients(zip(self.actor_gradients, self.network_params))

            self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

            self.saver = tf.train.Saver()
            self.last_num_epi = -1
    
    def create_actor(self, scope_name, is_training=True, reuse = tf.AUTO_REUSE):

        if is_training == True:
            is_training = self.is_training

        
        with tf.variable_scope(scope_name) as scope:

            inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)

            with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                                    normalizer_fn = slim.batch_norm,
                                    normalizer_params = {"is_training": is_training},
                                    reuse = reuse,
                                    scope = scope):
                fc1 = slim.fully_connected(inputs, 400, scope="fc1")
                fc2 = slim.fully_connected(fc1, 300, scope="fc2")
                out = slim.fully_connected(fc2, self.a_dim, scope="action", activation_fn=tf.nn.sigmoid, normalizer_fn=None, weights_initializer=tf.truncated_normal_initializer(0, 0.001))
                out = tf.reduce_mean(out, axis=0)
                out = tf.stack([out for _ in range(self.batch_size)])             
        return inputs, out
    
    def train(self, inputs, a_gradient, num_epi, sess):
        if num_epi%5 == 0 and num_epi!=self.last_num_epi:
            self.saver.save(sess, self.save_path)
#             print "Actor Saved"
            self.last_num_epi = num_epi
        sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.is_training: True
        })
        
    def decay_learning_rate(self, decay_value, sess):
        sess.run( self.learning_rate.assign( self.learning_rate* decay_value))
        
    def get_learning_rate(self, sess):
        return sess.run(self.learning_rate)

    def predict(self, inputs, sess):
        return sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.is_training: False
        })

    def predict_target(self, inputs, sess):
        return sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.is_training: False
        })

    def update_target_network(self, sess):
        sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class Critic(object):
    
    def __init__(self, graph, state_dim, action_dim, learning_rate, tau, gamma, save_path):
        self.graph = graph
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        self.tau = tau
        self.gamma = gamma
        self.save_path = save_path
        self.is_training = tf.placeholder(shape=[], dtype=tf.bool)
        
        with graph.as_default():
        
            self.inputs, self.action, self.out = self.create_critic("Critic")
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Critic/")

            # Target
            self.target_inputs, self.target_action, self.target_out = self.create_critic("Critic_target", is_training=False)
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Critic_target/")

            self.update_target_network_params = [self.target_network_params[i]
                                                 .assign(tf.multiply(self.network_params[i], self.tau) 
                                                         + tf.multiply(self.target_network_params[i], 1.-self.tau)) 
                                                 for i in range(len(self.target_network_params))]

            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

            self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="Critic/"))
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "Critic/")
            self.optimize = tf.train.AdamOptimizer(self.learning_rate)
            with tf.control_dependencies(self.update_ops):
                self.optimize = self.optimize.minimize(self.loss)

            self.action_grads = tf.gradients(self.out, self.action)

            self.saver = tf.train.Saver()
            self.last_num_epi = -1

    def create_critic(self, scope_name, is_training=True, reuse = tf.AUTO_REUSE):
        if is_training == True:
            is_training = self.is_training


        with tf.variable_scope(scope_name) as scope:

            s_inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
            a_inputs = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)

            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.01),
                                biases_regularizer=slim.l2_regularizer(0.01),
                                normalizer_fn = slim.batch_norm,
                                normalizer_params = {"is_training": is_training},
                                reuse = reuse,
                                scope = scope):
                fc1_1 = slim.fully_connected(s_inputs, 400, scope="fc1_1")
                fc1_2 = slim.fully_connected(a_inputs, 300, scope="fc1_2")
                fc2 = slim.fully_connected(fc1_1, 300, scope="fc2") + fc1_2
                out = slim.fully_connected(fc2, 1, scope="fc3", activation_fn=None, normalizer_fn=None, weights_initializer=tf.truncated_normal_initializer(0, 0.001), weights_regularizer=None)
                
        return s_inputs, a_inputs, out
    
    def train(self, inputs, action, predicted_q_value, num_epi, sess):
        if num_epi%5 == 0 and num_epi!=self.last_num_epi:
            self.saver.save(sess, self.save_path)
#             print "Critic Saved"
            self.last_num_epi = num_epi
        
        return sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.is_training: True
        })

    def predict(self, inputs, action, sess):
        return sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.is_training: False
        })

    def predict_target(self, inputs, action, sess):
        return sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.is_training: False
        })

    def action_gradients(self, inputs, actions, sess):
        return sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.is_training: False
        })

    def update_target_network(self, sess):
        sess.run(self.update_target_network_params)
        
    def decay_learning_rate(self, decay_value, sess):
        sess.run( self.learning_rate.assign( self.learning_rate* decay_value))
        
    def get_learning_rate(self):
        return self.sess.run(self.learning_rate)
    
        
        