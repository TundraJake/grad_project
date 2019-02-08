'''

Jacob McKenna
UAF Graduate Project
nn.py - Keras NN market prediction.

'''
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence, one_hot
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy.random as npr
import numpy as np
import nltk

# # from tensorflow.python.client import device_lib
# # print(device_lib.list_local_devices())


# from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())
# # config = ConfigProto( device_count = {'GPU': 1} )
# # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # keras.backend.set_session(sess) 

class Neural_Network():

	def __init__(self, input_shape):
		npr.seed(1)
		self.model = Sequential()
		self.input_shape_ = input_shape
		self.x_train_ = None
		self.y_train_ = None
		self.x_test_ = None

		self.__add_layers()

	def __add_layers(self):
		self.model.add(Dense(self.input_shape_, input_shape=(self.input_shape_,), activation='relu'))
		self.model.add(Dense(20, activation='relu'))
		self.model.add(Dense(14, activation='relu'))
		self.model.add(Dense(1, activation='sigmoid'))

	def get_x_training_set(self):
		return self.x_train_

	def get_y_training_set(self):
		return self.y_train_

	def load_data(self, x_train, y_train):
		self.x_train_ = x_train
		self.y_train_ = y_train

	def train(self):
		self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=1.0), metrics=['accuracy'])
		self.model.fit(self.x_train_, self.y_train_, epochs=75, batch_size=1)

	def evaulate(self):
		scores = self.model.evaluate(self.x_train_, self.y_train_, verbose=0)
		print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

	def write_to_json(self):
		# serialize model to JSON
		model_json = self.model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		
	def write_weights_to_h5(self):
		# serialize weights to HDF5
		self.model.save_weights("model.h5")
		print("Saved model to disk")

	def load_weights_from_h5(self):
		# Model needs to be built first
		self.model.load_weights("model.h5")

	def predict(self):
		print()