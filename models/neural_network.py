from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence, one_hot
from sklearn.preprocessing import MinMaxScaler
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import keras
from keras.models import Sequential
import matplotlib.pyplot as plt 
import numpy.random as npr
import numpy as np
import nltk

DIR = 'results/'
FFN_DIR = DIR + 'ffn/'

class Neural_Network_Base(object):

    def __init__(self, X, Y, X_test, Y_test, name, epochs, batch_size):

        self.name_ = name
        self.x_train_ = X
        print("X training shape: ", self.x_train_.shape)
        self.y_train_ = Y
        print("Y training shape: ", self.y_train_.shape)

        self.x_test_ = X_test
        self.y_test_ = Y_test
        print("X test shape: ", self.x_test_.shape)
        print("Y test shape: ", self.y_test_.shape)

        self.__epochs_ = epochs
        self.__batch_size_ = batch_size

        self.history_ = None

    def get_epochs(self):
        return self.__epochs_

    def get_batch_size(self):
        return self.__batch_size_

    def get_x_training_set(self):
        return self.x_train_

    def get_y_training_set(self):
        return self.y_train_

    def get_history(self):
        return self.history_.history

    def predict(self):
        scalar = MinMaxScaler()

        predictions = self.model.predict(self.x_test_)
        real_stock_price = self.y_test_
        # real_stock_price = scarar.inverse_transform(self.x_test_[:, 0])
        # predictions = scarar.inverse_transform(predictions)
        scalar.fit(predictions)
        predictions = scalar.inverse_transform(predictions)
        plt.plot(real_stock_price, color = 'black', label = 'Actual Price')
        plt.plot(predictions, color = 'green', label = 'Predicted Price')


        plt.title(self.name_ + ' Performance w/' + str(self.__epochs_) + ' epochs')
        plt.xlabel('Trading Day')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.savefig('data/results/' + self.name_)

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

    def plot_accuracy(self):
        plt.plot(self.history_.history['acc'])
        plt.plot(self.history_.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def load_model(self, filename):
        self.model.load_weights(filename)

    def write_loss_history_graph(self):
        hist = self.get_history()
        plt.plot(hist['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('data/results/' + self.name_ + '_loss.png')
        plt.clf()

    def evaluate(self):
        scores = self.model.evaluate(self.x_test_, self.y_test_)
        print("Model Metrics: ", self.model.metrics_names)
        print("Scores: ", scores)
        self.write_loss_history_graph()    