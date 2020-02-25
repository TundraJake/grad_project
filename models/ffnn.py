from models.neural_network import Neural_Network
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import keras
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

import numpy as np
np.random.seed(1) 

class FFNN(Neural_Network):

    def __init__(self, X, Y, X_test, Y_test, name):
        super().__init__(X, Y, X_test, Y_test, name)
        self.model = self.__build_model()

    def __build_model(self):
        model = Sequential()
        model.add(Dense(2048, input_dim = self.x_train_.shape[1]))
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(1))
        return model

    def train(self):
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.summary()
        self._checkpoint()
        self.history_ = self.model.fit(self.get_x_training_set(), self.get_y_training_set(), 
                        epochs=750, batch_size=5, verbose=1)

    def plot_loss_graph(self):
        plt.plot(self.history_.history['loss'])
        # plt.plot(self.history_.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # print("%s: %.2f%%" % (self.model.metrics_names[1], scores

    def write_history_to_file(self):
        numpy_loss_history = np.array(loss_history)
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

