'''

Jacob McKenna
UAF Graduate Project
rnn.py - Keras NN market prediction.

'''

from models.neural_network import Neural_Network
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
import keras
from keras.callbacks import ModelCheckpoint

class R_NN(Neural_Network):

    def __init__(self, X, Y, X_test, Y_test):
        super().__init__(X, Y, X_test, Y_test)

        self.model = self.__build_model()

    def __build_model(self):
        model = Sequential()
        model.add(SimpleRNN(input_shape = (self.x_train_.shape[1], 1)))
        model.add(Dropout(0.1))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dropout(0.1))
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.1))
        model.add(Dense(256))
        return model

    def write_loss_history_graph(self):
        hist = self.get_history()
        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        savefig('results/lstm_loss_01lr_sp500.png')
        plt.clf()

    def train(self):
        opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9,)
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self._checkpoint()
        self.history_ = self.model.fit(self.get_x_training_set(), self.get_y_training_set(), epochs=10, batch_size=30, verbose=1)
        # print(self.history_.history.keys())

    def evaluate(self):
        scores = self.model.evaluate(self.x_test_, self.y_test_)
        # summarize history for loss
        plt.plot(self.history_.history['loss'])
        # plt.plot(self.history_.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
    
    def create_and_load_model(self, filename):
        self.model.load_weights(filename)
		
    def write_history_to_file(self):
        numpy_loss_history = np.array(loss_history)
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

    def graph_accuracy(self):
        plt.plot(self.history_.history['acc'])
        # plt.plot(self.history_.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()