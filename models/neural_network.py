from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence, one_hot
from sklearn.preprocessing import MinMaxScaler
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import keras
from keras.models import Sequential
import matplotlib.pyplot as plt 
import numpy.random as npr
import numpy as np
import nltk


from sklearn.preprocessing import MinMaxScaler

from settings import *

class Neural_Network_Base:

    def __init__(self, training_set, testing_set, sym, name, exp, epochs, batch_size):

        self.__training_set = training_set
        self.__testing_set = testing_set

        self.training_scaler = MinMaxScaler()
        self.testing_scaler = MinMaxScaler()

        self.__fit_data(training_set, testing_set)

        self.__name_ = name
        self.__exp_ = exp
        self.__symbol = sym

        self.__epochs_ = epochs
        self.__batch_size_ = batch_size

        self.history_ = None
    
    def __fit_data(self, training_set, testing_set):
        self.set_training_set(self.training_scaler.fit_transform(self.get_training_set()))
        self.set_testing_set(self.testing_scaler.fit_transform(self.get_testing_set()))

    def set_training_set(self, data):
        self.__training_set = data

    def set_testing_set(self, data):
        self.__testing_set = data

    def get_training_set(self):
        return self.__training_set

    def get_testing_set(self):
        return self.__testing_set

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

    def get_x_test_set(self):
        return self.x_test_

    def get_y_test_set(self):
        return self.y_test_

    def get_history(self):
        return self.history_.history

    def scale_predictions(self, predictions):
        pred = MinMaxScaler()
        pred.min_, pred.scale_ = self.testing_scaler.min_[-1], self.testing_scaler.scale_[-1]
        
        return pred.inverse_transform(predictions)

    def plot_price_differences(self, predictions, actual):
        values = np.subtract(predictions[1:], actual[:-1])

        plt.plot(values, color='green')
        plt.title('Differences between Predictions and Actual Prices')
        plt.xlabel('Trading Day')
        plt.ylabel('Price Difference ($)')
        plt.savefig(self.get_save_figure_location() + '_diff')
        plt.clf()

    def predict(self):
        predictions = self.model.predict(self.get_x_test_set())
        predictions = self.scale_predictions(predictions)

        real_stock_prices = self.scale_predictions( self.y_test_.reshape(self.y_test_.shape[0], 1) )
        all_days = self.y_test_

        # Review data before final...
        plt.plot(real_stock_prices[:-2], color='blue', label='Actual Price')
        plt.plot(predictions[1:], color = 'red', label = 'Predicted Price')

        plt.title(self.get_symbol() + ' Performance w/' + str(self.__epochs_) + ' epochs')
        plt.xlabel('Trading Day')
        plt.ylabel('Price ($)')
        plt.legend()

        plt.savefig(self.get_save_figure_location())

        plt.clf()
        self.plot_price_differences(predictions, real_stock_prices)

    def summary(self):
        self.model.summary()

    def write_weights_to_h5(self):
        # serialize weights to HDF5
        self.model.save_weights(self.get_name() + "/_model.h5")
        print("Saved")

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

    def plot_model(self):
        plot_model(self.model, to_file=self.get_name() + '_' + self.get_experiment_name() + '.png', show_shapes=True)