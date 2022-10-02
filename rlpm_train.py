import argparse
import os

import pandas as pd
import numpy as np
import csv
import sys

from itertools import groupby

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from collections import Counter
import random
from rl_pattern_mining import *
from utils import *
import scipy
import multiprocessing as mp

slim = tf.contrib.slim
rnn = tf.contrib.rnn

session_config = tf.ConfigProto(log_device_placement=False)
session_config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser()
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-wOD", dest='wOD', action='store_true', help="If input the original data along with MPTS")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-initial_learning_rate", type=float, help="Set learning rate for training the classification model DEFAULT=0.0005", default=0.0005)
parser.add_argument("-actor_lr", type=float, help="Set learning rate for training the actor DEFAULT=0.00005", default=0.00005)
parser.add_argument("-critic_lr", type=float, help="Set learning rate for training the critic DEFAULT=0.0005", default=0.0005)
parser.add_argument("-decay_steps", type=int, help="Set exponential decay step DEFAULT=1000", default=1000)
parser.add_argument("-decay_rate", type=float, help="Set exponential decay rate DEFAULT=0.95", default=0.95)
parser.add_argument("-decay_lr_actor", type=float, help="Set decay rate the learning rate of the actor DEFAULT=0.965", default=0.965)
parser.add_argument("-decay_lr_critic", type=float, help="Set decay rate the learning rate of the critic DEFAULT=0.965", default=0.965)
parser.add_argument("-training_steps", type=int, help="Set max number of training epochs DEFAULT=8000", default=8000)
parser.add_argument("-seed", type=int, help="Set random seed", default=2599)
parser.add_argument("-exploration_prob", type=float, help="Initial probability of random exploration (1-p1-p2) in the behavioral policy", default=0.3)
parser.add_argument("-all_ones_prob", type=float, help="Initial probability of making action as a vector with all ones (p2) in the behavioral policy", default=0.6)
parser.add_argument("-exploration_prob_decay", type=float, help="Rate of decaying the probability of random exploration in each step", default=0.99)
parser.add_argument("-all_ones_prob_decay", type=float, help="Rate of decaying the probability of following the heuristic in each step", default=0.99)
parser.add_argument("-rl_reward_thres_for_decay", type=float, help="Threshold on cumulative reward to decay actor/critic learning rates DEFAULT=-0.01", default=-0.01)
parser.add_argument("-replay_buffer", type=int, help="Size of experience replay buffer for training actor and critic. Default to 10**4.", default=10**3)
parser.add_argument("-data", type=str, help="Datset name.", required=True)
parser.add_argument("-cluster", type=str, help="Total number of clusters in event clustering.", required=True)
parser.add_argument("-split_idx", type=int, help="Train/test split index.", required=True)
parser.add_argument("-nodeN", type=int, help="Num of LSTM nodes DEFAULT=256", default=256)
parser.add_argument("-weights", type=int, nargs='+', help="Number of nodes for hidden layers of classification model.", required=True)
parser.add_argument("-batch_size", type=int, help="Training batch size DEFAULT=256", default=256)

if __name__ == '__main__':

	args = parser.parse_args()
	if not args.no_gpu:
	    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
	    session_config = tf.ConfigProto(log_device_placement=False)
	    session_config.gpu_options.allow_growth = True
	else:
	    os.environ["CUDA_VISIBLE_DEVICES"] = ""
	    session_config = tf.ConfigProto(log_device_placement=False)

	# Set random seeds
	SEED = args.seed
	np.random.seed(SEED)
	tf.set_random_seed(SEED)
	random.seed(SEED)

	# Create directories to store checkpoints, logs and results
	if not os.path.exists("./saved_models"):
	    os.mkdir("saved_models")
	if not os.path.exists("./stats"):
	    os.mkdir("stats")
	if not os.path.exists("./stats/rl_log"):
	    os.mkdir("stats/rl_log")

    # Read in MPTS
	data = args.data
	cluster = args.cluster
	split_idx = args.split_idx
	all_ins_pat_df = pd.read_csv('./result/'+data+'_pattern_c'+cluster+'.csv')

	if args.wOD:

		original_df = pd.read_csv('./data/'+data+'/data_id_label.csv')
		original_df = original_df.rename(columns={'X'+str(i):'X'+str(i+all_ins_pat_df.shape[1]-4) for i in range(1, original_df.shape[1]-1)})
		original_df = original_df[[c for c in original_df.columns.values if 'X' in c]]
		original_df = (original_df - original_df.mean())/original_df.std()

		df = pd.concat([original_df, all_ins_pat_df], axis=1)
		df = df.fillna(0)

	else:

		df = all_ins_pat_df
		df = df.fillna(0)

	numeric_columns = [c for c in df.columns.values if 'X' in c]

	print('Imputed data contains any NaN?', df[numeric_columns].isnull().values.any())
	new_columns = ['visitID'] + numeric_columns

	X = []
	y = []
	lengths = []
	ins_num = len(df['visitID'].unique())
	column_num = df.shape[1] - 3
	var_column = ['X' + str(i) for i in range(0, column_num)]

	for ins in df['visitID'].unique():
	    ins_label = df[df['visitID'] == ins]['Label'].iloc[0]
	    X.append(df[df['visitID']==ins][var_column].values.tolist())
	    y.append(ins_label)
	    lengths.append(len(df[df['visitID']==ins]))

	print("mean length of seqs:", np.array(lengths).mean())
	print ('Number of instances in each class')
	print(Counter(y))
	X = np.array(X)
	y = np.array(y)

	class_num = len(Counter(y))
	max_length = int(np.array(lengths).mean())
	X = tf.compat.v1.keras.preprocessing.sequence.pad_sequences(
		X, maxlen=max_length, padding = 'pre', truncating='pre',dtype = 'float64'
	)
	X = np.reshape(X, (X.shape[0], max_length, X.shape[2])).astype(np.float32)
	print('X dim after padding:', X.shape)

	# Define train and test sets according to the train/test splits provided by UEA

	X_train = X[:split_idx]
	X_test = X[split_idx:]
	y_train = y[:split_idx]
	y_test = y[split_idx:]

    # Set hyper-parameters

	initial_learning_rate = args.initial_learning_rate
	decay_steps = args.decay_steps
	decay_rate = args.decay_rate
	weights = args.weights
	actor_lr = args.actor_lr
	critic_lr = args.critic_lr

	nodeN = args.nodeN
	batch_size = args.batch_size

	rl_reward_thres_for_decay = args.rl_reward_thres_for_decay

	training_steps = args.training_steps
	display_step = 10


    # Set the parameter for the Bernoulli distribution used for exploration
	explore_action_random_prob = .8

	# Create classification model

	graph = tf.Graph()

	def create_model(x, is_training=True, reuse=tf.AUTO_REUSE, graph=graph):
	    with graph.as_default():

	    	# Convert inputs into LSTM encodings
	        with tf.variable_scope("lstm", reuse=reuse) as scope:

	            x = tf.unstack(x, max_length, 1)
	            lstm_cell = rnn.BasicLSTMCell(nodeN, forget_bias=1.0, reuse=reuse)
	            outputs, state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	            # Now define the dense layers for classification
	            with slim.arg_scope([slim.fully_connected], 
	                                    activation_fn=tf.nn.relu,
	                                    weights_initializer=tf.truncated_normal_initializer(0., 0.00001),
	                                    biases_initializer=tf.zeros_initializer(),
	                                    weights_regularizer=slim.l2_regularizer(0.01),
	                                    biases_regularizer=slim.l2_regularizer(0.01),
	                                    normalizer_fn = slim.batch_norm,
	                                    normalizer_params = {"is_training": is_training},
	                                    reuse = reuse,
	                                    scope = scope):

	                if len(weights)==1:
	                    fc1 = slim.fully_connected(outputs[-1], weights[0], scope='fc1')
	                    logits = slim.fully_connected(
	                    	fc1,class_num,activation_fn=None, 
	                    	biases_initializer=None, biases_regularizer=None, 
	                    	weights_regularizer=None, normalizer_fn=None, scope='logits'
                    	)
	                else:
	                    fc1 = slim.fully_connected(outputs[-1], weights[0], scope='fc1')
	                    fc2 = slim.fully_connected(fc1, weights[1], scope='fc2')
	                    logits = slim.fully_connected(
	                    	fc2,class_num,activation_fn=None, biases_initializer=None, 
	                    	biases_regularizer=None, weights_regularizer=None, 
	                    	normalizer_fn=None, scope='logits'
                    	)

	                outs = tf.nn.sigmoid(logits)

	                return logits, outs, outputs[-1], state


	# Convert training/testing labels into one-hot encodings
	with tf.Session(config=session_config) as sess:
	    one_hot_labels_train = sess.run(slim.one_hot_encoding(y_train,class_num))
	    one_hot_labels_test = sess.run(slim.one_hot_encoding(y_test,class_num))

	print(slim.one_hot_encoding(y_test,class_num))

	# Set up the function that will be called by tf.data.Dataset to pull out data during training and validation

	def gen_train():
	    for i in range(len(X_train)):
	        yield X_train[i], one_hot_labels_train[i]    

	def gen_test():
	    for i in range(len(X_test)):
	        yield X_test[i], one_hot_labels_test[i]


	timesteps = len(X[0])
	dimension = len(X[0][0])

	with graph.as_default():

		# Create training dataset
	    dataset_train = tf.data.Dataset.from_generator(
	    	gen_train, (tf.float32, tf.float32), ([ timesteps, dimension],[ class_num])
    	).repeat(100000).shuffle(5000).batch(batch_size)
	    input_train, label_train = dataset_train.make_one_shot_iterator().get_next()

	    # Create validation dataset
	    dataset_test = tf.data.Dataset.from_generator(
	    	gen_test, (tf.float32, tf.float32), ([ timesteps, dimension],[ class_num])
    	).repeat(100000).batch(len(one_hot_labels_test))
	    input_test, label_test = dataset_test.make_one_shot_iterator().get_next()

	    # Placeholders for inputs and labels used for training
	    input_train_holder = tf.placeholder(shape=[batch_size, timesteps, dimension], dtype=tf.float32)
	    label_train_holder = tf.placeholder(shape=[batch_size, class_num], dtype=tf.float32)

	    # Placeholders will be used to validate the model
	    input_test_holder = tf.placeholder(shape=[len(one_hot_labels_test), timesteps, dimension], dtype=tf.float32)

	    # Build the computational graph for training
	    logit_train, pred_train, lstm_out_train, lstm_state_train = create_model(input_train_holder)

	    # Build the computational graph for validation, is_training=False will make sure inputs not to update 
	    # weights in batch normalization layers
	    logit_test, pred_test, _, _ = create_model(input_test_holder, is_training=False)

	    # Cross entropy loss for training
	    ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	    	labels=label_train_holder, logits=logit_train)
	    )

	    # Total loss for training = CE loss + regularization loss
	    loss = ce_loss + tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

	    # Setup learning rate and Adam gradient descent optimizer
	    lr = tf.train.exponential_decay(
	    	initial_learning_rate, tf.train.get_or_create_global_step(), 
	    	decay_steps=decay_steps, decay_rate=decay_rate
    	)
	    optimizer = tf.train.AdamOptimizer(lr)
	    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	    # Make sure batch normalization module is updated before the gradient steps
	    with tf.control_dependencies(update_ops):
	        train_op = optimizer.minimize(loss)


	    # Set up the paths that will be used to store model checkpoints, logs etc.
	    saver = tf.train.Saver()
	    if len(weights)>1:
	        file_appendix = "RLPM_BN_{:.6f}_{:.0f}_{:.2f}_{}_{}_{:.6f}_{:.6f}_{:.2f}".format(
	        	initial_learning_rate, decay_steps, decay_rate, weights[0], weights[1], 
	        	actor_lr, critic_lr, rl_reward_thres_for_decay
        	)
	    else:
	        file_appendix = "RLPM_BN_{:.6f}_{:.0f}_{:.2f}_{}_{}_{:.6f}_{:.6f}_{:.2f}".format(
	        	initial_learning_rate, decay_steps, decay_rate, weights[0], 0, 
	        	actor_lr, critic_lr, rl_reward_thres_for_decay
        	)
	
	with graph.as_default():
		# Define the element in computational graph that can plug in testing data 
		# and examine model performance on-the-fly, using labels in the format of one-hot encodings
	    accuracy = tf.reduce_mean(
	    	tf.cast(
	    		tf.equal(
	    			tf.argmax(logit_test, axis=1), 
	    			tf.argmax(tf.cast(one_hot_labels_test, tf.float32), axis=1)
    			), 
    		tf.float32)
    	)

	    # Define the element in computational graph that can display training accuracy on-the-fly
	    # using labels in the format of one-hot encodings
	    train_accuracy = tf.reduce_mean(
	    	tf.cast(
	    		tf.equal(
	    			tf.argmax(logit_train, axis=1), 
	    			tf.argmax(tf.cast(label_train_holder, tf.float32), axis=1)
    			), 
			tf.float32)
		)


	with graph.as_default():
		# Initiate actor and critic networks for RL
	    actor = Actor(
	    	graph=graph, state_dim=timesteps*dimension+nodeN*2, 
	    	action_dim=dimension, learning_rate=actor_lr, tau=0.001, 
	    	batch_size=batch_size, save_path="./saved_models/"+file_appendix+"/" + data+"_actor.ckpt"
    	)
	    critic = Critic(
	    	graph=graph, state_dim=timesteps*dimension+nodeN*2, 
	    	action_dim=dimension, learning_rate=critic_lr, tau=0.001, 
	    	gamma=0.99, save_path="./saved_models/"+file_appendix+ "/" + data+"_critic.ckpt"
    	)

	# Define the loss that examines the convergence of selected patterns (results shown in Appendix E)
	def binary_cross_entropy(yhat, y):
	    y = np.asarray(y)
	    yhat = np.asarray(yhat)
	    return -(y * np.log(yhat) + (1. - y) * np.log(1. - yhat)).mean()

	# Dummy variables that will be used to store intermediate results for visulization/logging
	prior_a = [0.] * dimension # added for visualization
	last_a_from_actor = None
	max_acc = 0.

	with tf.Session(config=session_config, graph=graph) as sess:
	    init = tf.global_variables_initializer()
	    sess.run(init)

	    # Set up the probability for exploration or taking as input all patterns
	    # as will be used in the exploration policy
	    EXPLORATION_RATE = args.exploration_prob
	    GUIDE_RATE = args.all_ones_prob
	    ep_reward = 0
	    ep_ave_max_q = 0

	    save_pattern_num = [] # added for visualization
	    save_pattern_bce = [] # added for visualization

	    # Get training data and labels for training, also get the data that will be used to 
	    # validate the model on-the-fly
	    data_in, label_in, data_in_test = sess.run([input_train, label_train, input_test])

	    # Get LSTM encodings
	    s_1 = sess.run(lstm_state_train, feed_dict = {input_train_holder:data_in, label_train_holder:label_in})

	    # Constitue the MDP state
	    s = np.concatenate([data_in.reshape(batch_size, -1), s_1[0], s_1[1]], axis=-1)

		# Empty lists defined to store episode rewards and Q-values to be logged later	
	    reward_list = []
	    ave_max_q_list = []

	    # Initiate experience replay buffer to train the RL agent
	    replay_buffer = ReplayBuffer(args.replay_buffer, random_seed=SEED)

	    # Initiate the target networks in DDPG
	    actor.update_target_network(sess)
	    critic.update_target_network(sess)

	    for step in range(training_steps):

	    	################## Exploration Policy to Determine MDP action #######################

	    	# Generate a random number for the exploration policy
	        rand_num = np.random.rand(1)

	        if rand_num <= EXPLORATION_RATE:
	        	# Take random actions
	            a = scipy.stats.bernoulli(explore_action_random_prob).rvs(dimension).astype(np.float32)
	            a = np.asarray([a for _ in range(batch_size)]).astype(np.float32)
	            a_for_inputs = np.copy(a)

	        elif rand_num <= GUIDE_RATE+EXPLORATION_RATE and rand_num > EXPLORATION_RATE:
	        	# Use all the pattern as inputs
	            a = np.ones((batch_size,dimension)).astype(np.float32)
	            a_for_inputs = np.copy(a)

	        else:
	        	# Use the patterns selected by the policy as inputs
	            a = actor.predict(s.reshape(-1,timesteps*dimension+nodeN*2), sess)
	            a = a.reshape(batch_size, dimension).astype(np.float32)
	            a_for_inputs = (a>0.5).astype(np.float32)
	            last_a_from_actor = np.copy(a_for_inputs[0])

	            save_pattern_num.append(last_a_from_actor.sum()) # added for visualization
	            save_pattern_bce.append(binary_cross_entropy(last_a_from_actor, prior_a))
	            prior_a = last_a_from_actor
            ##################[END] Exploration Policy to Determine MDP action [END]#######################

            # Now train with whatever inputs determined by the exploration policy
            # using the data and labels from the training set
	        (_, train_loss, train_ce_loss, logit_train_printed)  = sess.run(
	        	[train_op, loss, ce_loss, logit_train], 
	        	feed_dict={
		        	input_train_holder:data_in*a_for_inputs[0], 
		        	label_train_holder:label_in
	        	}
	        )

	        # Now get the training data and labels that will be used to update the classification model 
	        # in the next round
	        data_in, label_in, data_in_test= sess.run([input_train, label_train, input_test])

	        # Now constitute the MDP state s2
	        s2_1= sess.run(lstm_state_train, feed_dict = {input_train_holder:data_in*a_for_inputs[0], label_train_holder:label_in})
	        s2 = np.concatenate([data_in.reshape(batch_size, -1), s2_1[0], s2_1[1]], axis=-1)

	        # Set the reward as the negative of training loss
	        r = np.repeat(-train_loss, batch_size)

	        # Add the (s, a, s', r) tuple into experience replay buffer
	        replay_buffer.add_batch([list(i) for i in zip(s.reshape(-1,timesteps*dimension+nodeN*2),a.reshape(-1,dimension),r,s2.reshape(-1,timesteps*dimension+nodeN*2))])

	        if replay_buffer.size() > batch_size:
	        	# If we have enough data in the experience replay buffer, then train the RL agent (policy)
	            s_batch, a_batch, r_batch, s2_batch = replay_buffer.sample_batch(batch_size)

	            # Calculate targets
	            target_q = critic.predict_target(
	                s2_batch, actor.predict_target(s2_batch, sess), sess)

	            y_i = []
	            for k in range(batch_size):
	                y_i.append(r_batch[k] + critic.gamma * target_q[k])

	            # Update the critic given the targets
	            predicted_q_value, _ = critic.train(
	                s_batch, a_batch, np.reshape(y_i, (batch_size, 1)), step, sess)

	            ave_max_q = np.amax(predicted_q_value)
	            ave_max_q_list += [ave_max_q]

	            # Update the actor policy using the sampled gradient
	            a_outs = actor.predict(s_batch, sess)
	            grads = critic.action_gradients(s_batch, a_outs, sess)
	            actor.train(s_batch, grads[0], step, sess)

	            # Update target networks
	            actor.update_target_network(sess)
	            critic.update_target_network(sess)

	        # Now this MDP steps is finished, the state now should be set as s2
	        s = s2

	        # Record the reward we got in this step, which will be logged later
	        reward_list += [r[0]]

	        # If we have already trained for a few steps who not take a look at the progresses that have been made
	        # and log a few things
	        if step % display_step == 0 and step > 0 and last_a_from_actor is not None:

	        	# First decay the probabilities of taking exploration or all patterns as inputs in next a few steps
	        	# Since as the policy gets trained it already figures better ways to select patterns
	            EXPLORATION_RATE = EXPLORATION_RATE * args.exploration_prob_decay
	            GUIDE_RATE = GUIDE_RATE * args.all_ones_prob_decay

	            # Now print out the training losses, training accuracies and validation accuracies 
	            # to see where we've been up to
	            ce_loss_printed, loss_printed, acc, train_acc = sess.run([ce_loss, loss, accuracy, train_accuracy], feed_dict = {input_train_holder:data_in*last_a_from_actor, label_train_holder:label_in, input_test_holder:data_in_test*last_a_from_actor})

	            # Also decay the learning rate for training the RL policy 
	            if np.sum(reward_list[-display_step:]) >= args.rl_reward_thres_for_decay:
	                actor.decay_learning_rate(args.decay_lr_actor, sess)
	                critic.decay_learning_rate(args.decay_lr_critic, sess)

	            # Save the models
	            if acc > max_acc:
	                max_acc = acc
	                saver.save(sess,"./saved_models/" + file_appendix + "/" +data+ ".ckpt")
	                if not os.path.exists("./stats/rl_log/" + file_appendix  + "/"):
	                    os.mkdir("./stats/rl_log/" + file_appendix  + "/")
	                np.savetxt("./stats/rl_log/" + file_appendix  + "/" + data+ "_action.txt", last_a_from_actor)

	            # Print out the reward, losses and accuracies we obtained earlier
	            print("Step " + str(step) + ", Reward=" + str(np.sum(reward_list[-display_step:])) + ", Minibatch Loss= " + \
	                  "{:.4f},{:.4f}".format(ce_loss_printed,loss_printed) + ", Training Accuracy= " + \
	                  "{:.3f}".format(train_acc) + \
	                  ", Max Testing Accuracy= ", "{:6f}".format(max_acc) + \
	                  ", Max Q= ", "{:6f}".format(np.mean(ave_max_q_list[-display_step:])))

	            # Also log these metrics to a file
	            if not os.path.exists("./stats/rl_log/" + file_appendix  + "/"):
	                os.mkdir("./stats/rl_log/" + file_appendix  + "/")
	            with open("./stats/rl_log/" + file_appendix  + "/" + data+ ".txt", "a") as myfile:
	                myfile.write("Step " + str(step) + ", Reward=" + str(np.sum(reward_list[-display_step:])) + ", Minibatch Loss= " + "{:.4f}".format(loss_printed) + ", Training Accuracy= " + "{:.3f}".format(train_acc) + ", Max Testing Accuracy= " + "{:6f}".format(max_acc) + "\n")

	# Now we've done all the training

	# Save the logs that will be used for analyzing the convergence of the discriminative
	# patterns selected by RL
	np.save('./result/pattern_num/'+file_appendix+"_"+data+'.npy', save_pattern_num)
	np.save('./result/pattern_bce/'+file_appendix+"_"+data+'.npy', save_pattern_bce)








