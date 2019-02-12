'''

Jacob McKenna
UAF Graduate Project
nn.py - Keras NN market prediction.

'''

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence, one_hot
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt 
import numpy.random as npr
import numpy as np
import nltk



# from tensorflow.python.client import device_lib
# import tensorflow as tf
# print(device_lib.list_local_devices())
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(tf.test.is_gpu_available())
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

# from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())
# config = ConfigProto( device_count = {'GPU': 1} )
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# keras.backend.set_session(sess) 

class Neural_Network():

	def __init__(self, X, Y, X_test, Y_test):
		npr.seed(1)
		
		self.x_train_ = X
		self.y_train_ = Y

		self.x_test_ = X_test
		self.y_test_ = Y_test

		self.model = self.__add_lstm_90lr_sp500_layers()

		self.history_ = None

	def __built_model(self):
		model = Sequential()
		model.add(LSTM(units = 50, return_sequences = True, input_shape = (self.x_train_.shape[1], 1)))
		model.add(Dropout(0.2))
		model.add(LSTM(units = 50, return_sequences = True))
		model.add(Dropout(0.2))
		model.add(LSTM(units = 50, return_sequences = True))
		model.add(Dropout(0.2))
		model.add(LSTM(units = 50))
		model.add(Dropout(0.2))
		model.add(Dense(units = 1))
		return model


	def __add_lstm_90lr_sp500_layers(self):
		model = Sequential()
		model.add(LSTM(units = 50, return_sequences = True, input_shape = (self.x_train_.shape[1], 1)))
		model.add(Dropout(0.2))
		model.add(LSTM(units = 50, return_sequences = True))
		model.add(Dropout(0.2))
		model.add(LSTM(units = 50, return_sequences = True))
		model.add(Dropout(0.2))
		model.add(LSTM(units = 50))
		model.add(Dropout(0.2))
		model.add(Dense(units = 1))
		return model

	def get_x_training_set(self):
		return self.x_train_



	def get_y_training_set(self):
		return self.y_train_

	def get_history(self):
		return self.history_.history

	def write_loss_history_graph(self):
		hist = self.get_history()
		plt.plot(hist['loss'])
		plt.plot(hist['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		savefig('results/lstm_loss_01lr_sp500.png')
		plt.clf()


	def train(self):
		opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9,)
		self.model.compile(loss='mean_squared_error', optimizer='adam')
		self.__checkpoint()
		self.history_ = self.model.fit(self.get_x_training_set(), self.get_y_training_set(), epochs=40, batch_size=3000, verbose=1)
		# print(self.history_.history.keys())

	def __checkpoint(self):
		"""
		Saves data between epochs.
		"""
		checkpoint = ModelCheckpoint("results/best_weights_lstm_01lr_sp500.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		self.callbacks_list = [checkpoint]

	def evaluate(self):
		scores = self.model.evaluate(self.x_test_, self.y_test_)
		# summarize history for loss
		plt.plot(self.history_.history['loss'])
		# plt.plot(self.history_.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		# print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

	def create_and_load_model(self, filename):
		self.__add_lstm_90lr_sp500_layers()
		self.model.load_weights(filename)

	def write_to_json(self):
		# serialize model to JSON
		model_json = self.model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		
	def write_weights_to_h5(self):
		# serialize weights to HDF5
		self.model.save_weights("model.h5")
		print("Saved model to disk")

	def load_weights_from_h5(self, file_path):
		# Model needs to be built first
		self.model.load_weights(file_path)

	def predict(self):
		return self.model.predict(self.x_test_)
	
	def summary(self):
		self.model.summary()

	def write_history_to_file(self):
		numpy_loss_history = np.array(loss_history)
		np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
	# def write_accuracy(self):
		# plt.plot(self.history_.history['acc'])
		# plt.plot(self.history_.history['val_acc'])
		# plt.title('model accuracy')
		# plt.ylabel('accuracy')
		# plt.xlabel('epoch')
		# plt.legend(['train', 'test'], loc='upper left')
		# plt.show()