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

from sklearn.preprocessing import MinMaxScaler

from settings import *

class Neural_Network_Base(object):

    def __init__(self, X, Y, X_test, Y_test, sym, name, exp, epochs, batch_size):

        self.__name_ = name
        self.__exp_ = exp
        self.__symbol = sym
        self.x_train_ = X
        # print("X training shape: ", self.x_train_.shape)
        self.y_train_ = Y
        # print("Y training shape: ", self.y_train_.shape)

        self.x_test_ = X_test
        self.y_test_ = Y_test
        # print("X test shape: ", self.x_test_.shape)
        # print("Y test shape: ", self.y_test_.shape)

        self.__epochs_ = epochs
        self.__batch_size_ = batch_size

        self.history_ = None
        self.__min_max_scalar = MinMaxScaler()
        self.__fit_data()
    
    def __fit_data(self):
        data = np.load(POST_PROCESSING_DIR + self.get_symbol() + '/' + self.get_experiment_name() + '_data.npy')
        self.__min_max_scalar.fit(data)

    def get_symbol(self):
        return self.__symbol

    def get_experiment_name(self):
        return self.__exp_

    def get_save_location(self):
        return NN_RESULT_DIR + self.get_experiment_name() + '/' + self.get_symbol() + '/'

    def get_batch_size(self):
        return self.__batch_size_
        
    def get_name(self):
        return self.__name_

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
        predictions = self.model.predict(self.x_test_)
        real_stock_price = self.y_test_
        real_stock_price = self.__min_max_scalar.inverse_transform(real_stock_price)
        #all_days = np.concatenate([self.y_train_,self.y_test_])
        all_days = self.y_test_

        plt.plot(all_days, color='blue', label='Actual Price')
        plt.plot(range(len(all_days) - len(predictions), len(all_days)), predictions, color = 'red', label = 'Predicted Price')

        plt.title(self.get_name() + ' Performance w/' + str(self.__epochs_) + ' epochs')
        plt.xlabel('Trading Day')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.savefig(self.get_save_figure_location())
        plt.clf()

    def summary(self):
        self.model.summary()

    def write_weights_to_h5(self):
        # serialize weights to HDF5
        self.model.save_weights(self.get_name() + "/_model.h5")
        print("Saved model to disk")

    def load_weights_from_h5(self):
        # Model needs to be built first
        try:
            self.model.load_weights(self.get_name() + '/_model.h5')
        except:
            print("Could not find or load " + self.get_name())
            sys.exit()

    def load_model(self, filename):
        self.model.load_weights(filename)

    def get_save_figure_location(self):
        return self.get_save_location() + self.get_name()

    def plot_accuracy_graph(self):
        hist = self.get_history()
        plt.plot(hist['acc'])
        plt.plot(hist['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.get_save_figure_location() + '_acc.png')
        plt.clf()

    def plot_loss_history_graph(self):
        hist = self.get_history()
        plt.plot(hist['loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(self.get_save_figure_location() + '_loss.png')
        plt.clf()

    def evaluate(self):
        scores = self.model.evaluate(self.x_test_, self.y_test_)
        print("Model Metrics: ", self.model.metrics_names)
        print("Scores: ", scores)
        self.plot_loss_history_graph() 
        # self.plot_accuracy_graph()  