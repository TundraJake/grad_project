from models.lstmnn import LSTMNN
from models.ffnn import FFNN

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
import keras
from keras.callbacks import ModelCheckpoint

import sys
import os
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

def load_data(sym, exp):
    return POST_PROCESSING_DIR + sym + '/' + exp + '_data.npy'

def build_nn_name(sym, name):
    return sym + '_' + name

def build_experiment(exp):
    networks = []

    for sym in SYMBOLS:
        data = np.load(load_data(sym, exp))
        training_set, test_set = build_training_test_sets(data)

        networks.append(FFNN(training_set, test_set, sym, 'ffnn5', exp, 300, 10))
        networks.append(LSTMNN(training_set, test_set, sym, 'lstm4', exp, 500, 10))

        # networks.append(LSTMNN(training_set, test_set, sym, 'test', exp, 10, 10))

        train_networks(networks)
        networks = []

def run_experiments():
    for exp in EXPERIMENTS:
        build_experiment(exp)

def main():
    run_experiments() 
    
if __name__ == "__main__":
    main()
