from models.neural_network import Neural_Network_Base
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import keras
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

import numpy as np

class FFNN(Neural_Network_Base):

    def __init__(self, X, Y, X_test, Y_test, sym, name, exp, epochs, batch_size):
        super().__init__(X, Y, X_test, Y_test, sym, name, exp, epochs, batch_size)
        self.model = self.__build_model()

    def __build_model(self):
        model = Sequential()
        model.add(Dense(2048, input_dim = self.x_train_.shape[1]))
        print('input dimensions:', self.x_train_.shape[1])
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(1))
        return model

    def train(self):    
        DISPLAY_TRAINING_PROGRESS = 2
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.summary()
        self.history_ = self.model.fit(self.get_x_training_set(), self.get_y_training_set(), 
                        epochs=self.get_epochs(), batch_size=self.get_batch_size(), verbose=DISPLAY_TRAINING_PROGRESS)

    def write_history_to_file(self):
        numpy_loss_history = np.array(loss_history)
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

