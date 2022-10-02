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

SEED = 2599

np.random.seed(SEED)
tf.set_random_seed(SEED)
random.seed(SEED)

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
parser.add_argument("-rl_reward_thres_for_decay", type=float, help="Threshold on cumulative reward to decay actor/critic learning rates DEFAULT=-0.01", default=-0.01)
parser.add_argument("-seed", type=int, help="Set random seed", default=2599)
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

	SEED = args.seed
	np.random.seed(SEED)
	tf.set_random_seed(SEED)
	random.seed(SEED)

	if not os.path.exists("./saved_models"):
	    os.mkdir("saved_models")
	if not os.path.exists("./stats"):
	    os.mkdir("stats")
	if not os.path.exists("./stats/rl_log"):
	    os.mkdir("stats/rl_log")

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
	X = tf.compat.v1.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_length, padding = 'pre', truncating='pre',dtype = 'float64')
	X = np.reshape(X, (X.shape[0], max_length, X.shape[2])).astype(np.float32)
	print('X dim after padding:', X.shape)

	# define train and test sets

	X_train = X[:split_idx]
	X_test = X[split_idx:]
	y_train = y[:split_idx]
	y_test = y[split_idx:]


	initial_learning_rate = args.initial_learning_rate
	decay_steps = args.decay_steps
	decay_rate = args.decay_rate
	weights = args.weights
	actor_lr = args.actor_lr
	critic_lr = args.critic_lr

	nodeN = args.nodeN
	batch_size = args.batch_size

	rl_reward_thres_for_decay = args.rl_reward_thres_for_decay # trainable

	graph = tf.Graph()

	def create_model(x, is_training=True, reuse=tf.AUTO_REUSE, graph=graph):
	    with graph.as_default():
	        with tf.variable_scope("lstm", reuse=reuse) as scope:

	            x = tf.unstack(x, max_length, 1)
	            lstm_cell = rnn.BasicLSTMCell(nodeN, forget_bias=1.0, reuse=reuse)
	            outputs, state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
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
	                    logits = slim.fully_connected(fc1,class_num,activation_fn=None, biases_initializer=None, biases_regularizer=None, weights_regularizer=None, normalizer_fn=None, scope='logits')
	                else:
	                    fc1 = slim.fully_connected(outputs[-1], weights[0], scope='fc1')
	                    fc2 = slim.fully_connected(fc1, weights[1], scope='fc2')
	                    logits = slim.fully_connected(fc2,class_num,activation_fn=None, biases_initializer=None, biases_regularizer=None, weights_regularizer=None, normalizer_fn=None, scope='logits')
	                outs = tf.nn.sigmoid(logits)

	                return logits, outs, outputs[-1], state


	with tf.Session(config=session_config) as sess:
	    one_hot_labels_train = sess.run(slim.one_hot_encoding(y_train,class_num))
	    one_hot_labels_test = sess.run(slim.one_hot_encoding(y_test,class_num))

	print(slim.one_hot_encoding(y_test,class_num))

	def gen_train():
	    for i in range(len(X_train)):
	        yield X_train[i], one_hot_labels_train[i]    

	def gen_test():
	    for i in range(len(X_test)):
	        yield X_test[i], one_hot_labels_test[i]


	timesteps = len(X[0])
	dimension = len(X[0][0])

	with graph.as_default():

	    dataset_test = tf.data.Dataset.from_generator(gen_test, (tf.float32, tf.float32), ([ timesteps, dimension],[ class_num])).repeat(100000).batch(len(one_hot_labels_test))
	    input_test, label_test = dataset_test.make_one_shot_iterator().get_next()

	    input_test_holder = tf.placeholder(shape=[len(one_hot_labels_test), timesteps, dimension], dtype=tf.float32)

	    logit_test, pred_test, _, _ = create_model(input_test_holder, is_training=False)

	    saver = tf.train.Saver()
	    if len(weights)>1:
	        file_appendix = "RLPM_BN_{:.6f}_{:.0f}_{:.2f}_{}_{}_{:.6f}_{:.6f}_{:.2f}".format(initial_learning_rate, decay_steps, decay_rate, weights[0], weights[1], actor_lr, critic_lr, rl_reward_thres_for_decay)
	    else:
	        file_appendix = "RLPM_BN_{:.6f}_{:.0f}_{:.2f}_{}_{}_{:.6f}_{:.6f}_{:.2f}".format(initial_learning_rate, decay_steps, decay_rate, weights[0], 0, actor_lr, critic_lr, rl_reward_thres_for_decay)
	with graph.as_default():
	    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit_test, axis=1), tf.argmax(tf.cast(one_hot_labels_test, tf.float32), axis=1)), tf.float32))

	def binary_cross_entropy(yhat, y):
	    y = np.asarray(y)
	    yhat = np.asarray(yhat)
	    return -(y * np.log(yhat) + (1. - y) * np.log(1. - yhat)).mean()


	prior_a = [0.] * dimension # added for visualization
	last_a_from_actor = None
	max_acc = 0.

	## Eval
	with tf.Session(config=session_config, graph=graph) as sess:
	    saver.restore(sess, os.path.join("./saved_models", file_appendix, args.data+'.ckpt'))
	    last_action = np.loadtxt("./stats/rl_log/"+file_appendix+ "/" + args.data+ "_action.txt")
	    data_in_test = sess.run(input_test)
	    print "Accuracy: ", sess.run(accuracy, feed_dict={input_test_holder:data_in_test*last_action})









