from models.lstmnn import LSTMNN
from models.ffnn import FFNN

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import keras
from keras.callbacks import ModelCheckpoint

import sys

import numpy as np

from settings import * 

# Reproducible
np.random.seed(1) 

def train_networks(networks):
    for network in networks:
        network.train()
        network.evaluate()
        network.summary()
        network.predict()

def calculate_upper_num_input_bound(num_rows):
    return int(np.floor(0.8*num_rows))

def build_training_test_sets(data):
    num_rows = data.shape[0]
    TRAIN_SET_BEGIN = 0
    TRAIN_SET_END = calculate_upper_num_input_bound(num_rows)

    TEST_SET_BEGIN = TRAIN_SET_END
    TEST_SET_END = num_rows

    training_set = data[TRAIN_SET_BEGIN:TRAIN_SET_END]
    test_set = data[np.arange(TEST_SET_BEGIN, TEST_SET_END)]

    return training_set, test_set

def build_FFNN(training_set, test_set, name, epochs, batch_size):
    X_train = training_set[:, 0:-1]
    # print(X_train)
    # print("The shape is now: ", X_train.shape)
    Y_train = training_set[:, -1]
    # print(Y_train)
    # print('Y_train', Y_train)

    X_test = test_set[:, 0:-1]
    Y_test = test_set[:, -1]
    # print('Y_test:', Y_test)

    return FFNN(X_train, Y_train, X_test, Y_test, name, epochs, batch_size)

def build_LSTM(training_set, test_set, symbol, epochs, batch_size):
    X_train = training_set[:, 0:-1]
    # reshape input to be 3D [samples, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]) )
    print(X_train.shape)
    Y_train = training_set[:, -1]

    X_test = test_set[:, 0:-1]
    # reshape input to be 3D [samples, timesteps, features]
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]) )
    Y_test = test_set[:, -1]

    return LSTMNN(X_train, Y_train, X_test, Y_test, symbol, epochs, batch_size)

def build_and_train_nns():
    symbol = 'AAPL'
    data = np.load(POST_PROCESSING_DIR + symbol + '/' + 'data.npy')
    training_set, test_set = build_training_test_sets(data)

    networks = []
#     networks.append(build_FFNN(training_set, test_set, 'ffnn0', 5, 5))
#     networks.append(build_FFNN(training_set, test_set, 'ffnn1', 50, 5))
#     networks.append(build_FFNN(training_set, test_set, 'ffnn2', 250, 5))
#     networks.append(build_FFNN(training_set, test_set, 'ffnn3', 400, 5))
    networks.append(build_LSTM(training_set, test_set, 'lstm0', 10, 5))
    networks.append(build_LSTM(training_set, test_set, 'lstm1', 50, 5))
    networks.append(build_LSTM(training_set, test_set, 'lstm2', 100, 5))
    print("wut")
    train_networks(networks)

def main():
    build_and_train_nns() 
    

if __name__ == "__main__":
    main()
