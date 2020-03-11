from models.neural_network import Neural_Network_Base
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import keras
from keras.callbacks import ModelCheckpoint

import numpy as np

EPOCHS = 200

class LSTMNN(Neural_Network_Base):

    def __init__(self, X, Y, X_test, Y_test, name):
        super().__init__(X, Y, X_test, Y_test, name)

        self.model = self.__build_model()

    # input_dim = self.x_train_.shape[1]
    def __build_model(self):
        model = Sequential()
        print(self.x_train_.shape)
        model.add(LSTM(units=2048, return_sequences = True,  input_shape=(self.x_train_.shape[1], 1) ))
        model.add(LSTM(1024, return_sequences = True))
        model.add(LSTM(512, return_sequences = True))
        model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(1, activation = 'relu'))
        return model

    def train(self):
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self._checkpoint()
        self.history_ = self.model.fit(self.get_x_training_set(), self.get_y_training_set(), epochs=EPOCHS, batch_size=5, verbose=1)
        # print(self.history_.history.keys())
		
    def write_history_to_file(self):
        numpy_loss_history = np.array(loss_history)
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

    def plot_accuracy_graph(self):
        plt.plot(self.history_.history['acc'])
        # plt.plot(self.history_.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
    