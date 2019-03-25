'''

Jacob McKenna
UAF Graduate Project
neural_network.py - Keras NN market prediction.

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


# For reproducible results
npr.seed(1)

class Neural_Network(object):

    def __init__(self, X, Y, X_test, Y_test):

        self.x_train_ = X
        self.y_train_ = Y

        self.x_test_ = X_test
        self.y_test_ = Y_test

        self.history_ = None

    def print_summary(self):
        try:
            self.model.summary()
        except:
            
    def get_x_training_set(self):
        return self.x_train_

    def get_y_training_set(self):
        return self.y_train_

    def get_history(self):
        return self.history_.history

    def predict(self):
        return self.model.predict(self.x_test_)
	
    def summary(self):
        self.model.summary()

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

    def _checkpoint(self):
        """
        Saves data between epochs.
        """
        checkpoint = ModelCheckpoint("results/best_weights_lstm_01lr_sp500.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]