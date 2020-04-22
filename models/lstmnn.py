from models.neural_network import Neural_Network_Base
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import keras
from keras.callbacks import ModelCheckpoint

import numpy as np

class LSTMNN(Neural_Network_Base):

    def __init__(self, training_set, testing_set, sym, name, exp, epochs, batch_size):
        Neural_Network_Base.__init__(self, training_set, testing_set, sym, name, exp, epochs, batch_size)

        self.x_train_ = self.get_training_set()[:, 0:-1]
        self.x_train_ = np.reshape(self.x_train_, (self.x_train_.shape[0], 1, self.x_train_.shape[1]) )
        self.y_train_ = self.get_training_set()[:, -1]

        self.x_test_ = self.get_testing_set()[:, 0:-1]
        self.x_test_ = np.reshape(self.x_test_, (self.x_test_.shape[0], 1, self.x_test_.shape[1]) )
        self.y_test_ = self.get_testing_set()[:, -1]

        self.model = self.__build_model()

    def __build_model(self):
        model = Sequential()
        shape = (self.x_train_.shape[1], self.x_train_.shape[2])
        model.add(LSTM(units=1024, return_sequences=True, input_shape=shape ))
        model.add(LSTM(512))
        model.add(Dense(1))
        return model

    def train(self):
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        print(self.get_x_training_set().shape)
        self.history_ = self.model.fit(self.get_x_training_set(), 
                                        self.get_y_training_set(), 
                                        epochs=self.get_epochs(), 
                                        batch_size=self.get_batch_size(), 
                                        verbose=1)
		
    def write_history_to_file(self):
        numpy_loss_history = np.array(loss_history)
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=',')
