from models.neural_network import Neural_Network_Base
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import keras
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

import numpy as np

class FFNN(Neural_Network_Base):

    def __init__(self, training_set, testing_set, sym, name, exp, epochs, batch_size):
        Neural_Network_Base.__init__(self, training_set, testing_set, sym, name, exp, epochs, batch_size)
        
        self.x_train_ = self.get_training_set()[:, 0:-1]
        self.y_train_ = self.get_training_set()[:, -1]
        self.x_test_ = self.get_testing_set()[:, 0:-1]
        self.y_test_ = self.get_testing_set()[:, -1]

        self.model = self.__build_model()

    def __build_model(self):
        model = Sequential()
        model.add(Dense(2048, input_dim = self.x_train_.shape[1]))
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def train(self):    
        DISPLAY_TRAINING_PROGRESS = 2
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.summary()
        self.history_ = self.model.fit(self.get_x_training_set(), self.get_y_training_set(), 
                        epochs=self.get_epochs(), batch_size=self.get_batch_size(), verbose=DISPLAY_TRAINING_PROGRESS, 
                        validation_data=(self.get_x_test_set(), self.get_y_test_set()))

    def write_history_to_file(self):
        numpy_loss_history = np.array(loss_history)
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

