'''

Jacob McKenna
UAF Graduate Project
main.py - Starting point of the entire project.

'''
from neural_network import *
from online_data_fetcher import * 
import pandas as pd

# from sklearn.feature_extraction import DictVectorizer
# vec = DictVectorizer()


def main():

    # X = np.array([[1,1,1,1,1,1,0,0,0,0,1,1],
    #                 [1,1,1,1,1,1,0,0,0,0,1,1],
    #                 [1,1,1,1,1,1,0,0,0,0,1,1],
    #                 [1,1,1,0,1,1,0,0,0,0,1,1],
    #                 [1,0,1,1,1,1,0,0,0,0,1,1],
    #                 [1,1,0,1,0,1,0,0,0,0,1,1],
    #                 [1,1,1,1,0,1,0,0,0,0,1,1]])

    # Y = np.array([1,0,0,1,1,0,1,])

    # X = encoded_docs




    data = pd.read_csv('data/numerical/set_1_stocks.csv')

    data = data.drop(['DATE'], 1)
    n = data.shape[0] 
    p = data.shape[1]

    data = data.values

    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n

    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    # Scale data
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)

    X = data_train[:, 1:]
    Y = data_train[:, 0]

    X_test = data_test[:, 1:]
    Y_test = data_test[:, 0]

    print("X Shape: ", X.shape[1])
    example_nn = Neural_Network(X.shape[1])

    example_nn.summary()
    example_nn.set_data(X, Y, X_test, Y_test)

    example_nn.train()

if __name__ == "__main__":
    main()



