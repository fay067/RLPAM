The repo is for paper: A Reinforcement Learning-Informed Pattern Mining Framework for
Multivariate Time Series Classification

This folder contains the code to (1) map multi-variate time series (MTS) to multi-pattern time series (MPTS) and (2) train the classifier using MPTS as inputs. We provide the generated MPTS and classifier checkpoints obtained from the ERing, MotorImagery and FingerMovements UEA datasets for checking reproducibility.

Folders:
data -- contains the example: ERing, MotorImagery and FingerMovements UEA datasets
results -- contains pre-converted MPTS for the three datasets in above
saved_models -- stores (pre-trained) RLPM checkpoints that can be loaded as classification models
stats -- stores the actions (selected discriminative patterns) generated by the RL module
gen_mpts -- contains the code for (univariate discrete sequences) UDS formulation and MPTS generation

############################################
(1) Convert MTS to MPTS 
############################################

1. Build ./gen_mpts/MGFSM_HOME following the readme file in that folder
Dependencies: Java JDK 1.6, Maven 3.2

2. Excecute all the the jupyter notebook MPTS_example.ipynb in ./gen_mpts/
Dependencies:
Python 3
numpy=1.19.2
pandas=1.1.3
csv=1.0
scipy=1.5.2
matplotlib=3.3.2
sklearn=0.23.2



############################################
(2) Train/Evaluate the Classification Model 
	Note that the code in this part can be executed stand-alone as all the pre-converted MPTS are saved under foler ./results.
############################################

Dependencies:
Python 2.7 / Python 3
tensorflow 1.15.0
scikit-learn 0.24.0
pandas 0.24.2
numpy 1.16.6
scipy 1.2.1


To load the pre-trained classifier checkpoints on MotorImagery/FingerMovements/ERing:

	python rlpm_eval.py -data MotorImagery -cluster 13 -split_idx 278 -initial_learning_rate 0.0005 -decay_steps 500 -decay_rate 0.95 -weights 512 256 -actor_lr 0.001 -critic_lr 0.0001 -wOD

	python rlpm_eval.py -data FingerMovements -split_idx 316 -cluster 12 -nodeN 256 -weights 128 -initial_learning_rate 0.0004 -decay_steps 1000 -decay_rate 0.95 -actor_lr 0.0005 -critic_lr 0.00001 -rl_reward_thres_for_decay -0.05 -wOD

	python rlpm_eval.py -data ERing -cluster 14 -split_idx 30 -initial_learning_rate 0.0001 -decay_steps 1000 -decay_rate 0.95 -actor_lr 0.0005 -critic_lr 0.0001 -nodeN 256 -weights 128 -rl_reward_thres_for_decay -0.1

To train a classifier on MotorImagery/FingerMovements/ERing using rlpm from scratch:

	python rlpm_train.py -data MotorImagery -cluster 13 -split_idx 278 -initial_learning_rate 0.0005 -decay_steps 500 -decay_rate 0.95 -weights 512 256 -actor_lr 0.001 -critic_lr 0.0001 -wOD

	python rlpm_train.py -data FingerMovements -split_idx 316 -cluster 12 -nodeN 256 -weights 128 -initial_learning_rate 0.0004 -decay_steps 1000 -decay_rate 0.95 -actor_lr 0.0005 -critic_lr 0.00001 -rl_reward_thres_for_decay -0.05 -wOD

	python rlpm_train.py -data ERing -cluster 14 -split_idx 30 -initial_learning_rate 0.0001 -decay_steps 1000 -decay_rate 0.95 -actor_lr 0.0005 -critic_lr 0.0001 -nodeN 256 -weights 128 -rl_reward_thres_for_decay -0.1


****************************************************************************************************
----------------------------------------------------------------------------------------------------
Due to the size limitation of the files that can be uploaded to Github, we do not provide the data, pre-converted MPTS and pre-trained checkpoints here. Please download the full version (1.3GB after de-compression) from https://drive.google.com/file/d/1OrjeVNtG32Llf2ag9QBLtOJvSOG9k81N/view?usp=sharing to access the data, pre-converted MPTS and pre-trained models.
----------------------------------------------------------------------------------------------------
****************************************************************************************************

