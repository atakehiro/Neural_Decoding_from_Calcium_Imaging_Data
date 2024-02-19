#!/usr/bin/env python
# coding: utf-8

# Copyright 2021 Takehiro Ajioka

# ### Enviroment
# 
# Python 3.6
# Anaconda
# tensorflow-gpu==1.15.0
# keras==2.3.1
# shap==0.36.0
# tfdeterminism==0.3.0

# In[1]:

def Learn_model_region(trainX, validX, trainY, validY, ModelType, ver, REGION):
	# Ignore warning
	import os
	import tensorflow as tf
	import logging
	import warnings
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	warnings.simplefilter(action='ignore', category=FutureWarning)
	warnings.simplefilter(action='ignore', category=Warning)
	tf.get_logger().setLevel('INFO')
	tf.autograph.set_verbosity(0)
	tf.get_logger().setLevel(logging.ERROR)


	# In[3]:


	import numpy
	import pandas
	import matplotlib.pyplot as plt
	from scipy import io
	from tensorflow.keras import layers, losses, optimizers, Input
	from tensorflow.keras.models import Sequential, load_model
	from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
	from tensorflow.keras.layers import Dense, Dropout, Flatten
	from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, ConvLSTM2D, Bidirectional
	from sklearn.metrics import roc_curve
	from sklearn.metrics import auc, roc_auc_score


	# In[4]:


	# Clear session
	from keras import backend as K
	K.clear_session()


	# In[5]:


	# Set Seed
	SEED = 123

	import os
	os.environ['PYTHONHASHSEED'] = str(SEED)
	os.environ['TF_DETERMINISTIC_OPS'] = 'true'
	os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

	import numpy as np
	import tensorflow as tf
	import random as rn
	np.random.seed(SEED)
	rn.seed(SEED)
	tf.set_random_seed(SEED)

	from keras import backend as K
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	K.set_session(sess)

	from tfdeterminism import patch
	patch()


	# In[7]:


	plt.rcParams["font.size"] = 18


	# # Deep Learning

	# In[16]:

	# Transpose input
	input_train = trainX.transpose(0,2,1)
	input_valid = validX.transpose(0,2,1)

	# In[17]:

	# Callback Setting
	model_file_path = f'models_N20/{ModelType}/{REGION}/best_model_{ver}.h5'
	modelCheckpoint = ModelCheckpoint(filepath = model_file_path,
		                          monitor='val_loss',
		                          verbose=0,
		                          save_best_only=True,
		                          save_weights_only=False,
		                          mode='min',
		                          save_freq="epoch")


	# In[18]:
	output_dim = 32 #This is determined by the complexity of the model.
	if ModelType == 'LSTM':
		RNN = LSTM(output_dim)
	elif ModelType == 'GRU':
		RNN = GRU(output_dim)
	elif ModelType == 'RNN':
		RNN = SimpleRNN(output_dim)
	elif ModelType == 'BiLSTM':
		RNN = Bidirectional(LSTM(output_dim))
	elif ModelType == 'BiGRU':
		RNN = Bidirectional(GRU(output_dim))
	elif ModelType == 'BiRNN':
		RNN = Bidirectional(SimpleRNN(output_dim))

	# Create and fit the RNN network
	model = Sequential()
	model.add(Input(shape=(trainX.shape[2], trainX.shape[1])))
	model.add(RNN)
	model.add(Dense(1, activation='sigmoid'))
	# model.summary()
	model.compile(loss=losses.BinaryCrossentropy(label_smoothing=0.001), optimizer='adam', metrics=['accuracy'])
	history = model.fit(x=input_train, y=trainY, validation_data=(input_valid, validY),
		            epochs=30, batch_size=256, verbose=0, callbacks=[modelCheckpoint])
	return history
# 	In[19]:
# 	# Plot learning result
# 	acc = history.history['acc']
# 	val_acc = history.history['val_acc']
# 	loss = history.history['loss']
# 	val_loss = history.history['val_loss']
# 	epochs = range(len(acc))
# 	plt.plot(epochs, loss, 'bo', label='Training loss')
# 	plt.plot(epochs, val_loss, 'b', label='Validation loss')
# 	plt.title('Training and validation loss')
# 	plt.xlabel("Epoch")
# 	plt.ylabel("Loss")
# 	plt.legend()
# 	plt.figure()
# 	plt.plot(epochs, acc, 'bo', label='Training accuracy')
# 	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# 	plt.title('Training and validation accuracy')
# 	plt.xlabel("Epoch")
# 	plt.ylabel("Accuracy")
# 	plt.legend()
# 	plt.show()


