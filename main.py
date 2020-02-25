from models.lstmnn import LSTMNN
from models.ffnn import FFNN


from models.neural_network import Neural_Network
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import keras
from keras.callbacks import ModelCheckpoint

import sys

import numpy as np

# Reproducible
np.random.seed(1) 

def train_networks(networks):
    for network in networks:
        network.train()
        network.evaluate()
        network.summary()
        network.write_weights_to_h5()
        network.load_weights_from_h5()
        network.predict()


def build_and_train_nns():

    networks = []
    data = np.load('data/preprocessed/prepped_data_daily_time_series_full_set_0.npy')

    n = data.shape[0]
    train_start = 0
    train_end = int(np.floor(0.8*n))

    test_start = train_end
    test_end = n

    data_train = data[train_start:train_end]
    data_test = data[np.arange(test_start, test_end)]

    # Data separation and shaping
    print("All Training Data ", data_train[:, 0])
    X = data_train[:, 0:]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y = data_train[:, 0]
    print('Prediction values: ', Y)
    X_test = data_test[:, 0:]
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print("X Testing set: ", X_test)
    Y_test = data_test[:, 0]
    print('Y Testing set: ', Y_test)
    
    networks.append(LSTMNN(X, Y, X_test, Y_test, 'LSMT_1'))
    
    # TODO: Get FFNN finished by adjusting the input size and determining what is wrong with FFNN not making any good result
    # despite the fact it's given the answer.
    #networks.append(FFNN(X, Y, X_test, Y_test, "FFNN_1"))

    train_networks(networks)


def main():
    # test_stuff()
    build_and_train_nns() 
    


if __name__ == "__main__":
    main()



