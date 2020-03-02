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

DIR = 'results/'
FFN_DIR = DIR + 'ffn/'

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
# For reproducible results
npr.seed(1)

class Neural_Network(object):

    def __init__(self, X, Y, X_test, Y_test, name):

        self.name_ = name
        self.x_train_ = X
        print("X training shape: ", self.x_train_.shape)
        self.y_train_ = Y
        print("Y training shape: ", self.y_train_.shape)

        self.x_test_ = X_test
        self.y_test_ = Y_test
        print("X test shape: ", self.x_test_.shape)
        print("Y test shape: ", self.y_test_.shape)

        self.history_ = None

    def get_x_training_set(self):
        return self.x_train_

    def get_y_training_set(self):
        return self.y_train_

    def get_history(self):
        return self.history_.history

    def predict(self):
        predictions = self.model.predict(self.x_test_)
        print("The Predictions: ", predictions)
        real_stock_price = self.x_test_[:, 0]
        print(len(self.x_test_))
        print(self.x_test_)
        predicted_stock_price = predictions

        ### TODO: Stop hardcoding labels and provide labels.
        plt.plot(real_stock_price, color = 'black', label = '')
        plt.plot(predicted_stock_price, color = 'green', label = '')
        plt.title('')
        plt.xlabel('')
        plt.ylabel('')
        plt.legend()
        plt.show()

    def summary(self):
        self.model.summary()

    def write_to_json(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write('results/' + model_json)

    def write_weights_to_h5(self):
        # serialize weights to HDF5
        self.model.save_weights(self.name_ + "_model.h5")
        print("Saved model to disk")

    def load_weights_from_h5(self):
        # Model needs to be built first
        self.model.load_weights(self.name_ + '_model.h5')

    def _checkpoint(self):
        checkpoint = ModelCheckpoint("", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]

    def plot_accuracy(self):
        plt.plot(self.history_.history['acc'])
        plt.plot(self.history_.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def write_loss_history_graph(self):
        hist = self.get_history()
        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        savefig('results/' + self.name_ + 'loss.png')
        plt.clf()

    def load_model(self, filename):
        self.model.load_weights(filename)

    def write_loss_history_graph(self):
        hist = self.get_history()
        plt.plot(hist['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('results/' + self.name_ + '_loss.png')
        plt.clf()

    def evaluate(self):
        scores = self.model.evaluate(self.x_test_, self.y_test_)
        print("Model Metrics: ", self.model.metrics_names)
        print("Scores: ", scores)
        self.write_loss_history_graph()    