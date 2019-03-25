'''

Jacob McKenna
UAF Graduate Project
main.py - Starting point of the entire project.

'''

from models.lstm_nn import LSTM_NN


from online_data_fetcher import * 
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
# create CountVectorizer object
vectorizer = CountVectorizer()

# from sklearn.feature_extraction import DictVectorizer
# vec = DictVectorizer()

from sqlalchemy import create_engine
engine = create_engine('postgresql://stock:money@localhost:5432/stock_market_data')


def lstm_sp500_model():
    data = pd.read_csv('data/numerical/sp500.csv')
    # data = pd.read_sql_query("select * from \"AAPL_ticker\"", con=engine)
    # print(type(data))
    data = data.drop(['DATE'], 1)
    
    n = data.shape[0] 

    # Scale data
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder as le
    scaler = MinMaxScaler(feature_range = (0, 1))
    data = data.values
    data = scaler.fit_transform(data)

    train_start = 0
    train_end = int(np.floor(0.8*n))

    test_start = train_end
    test_end = n

    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    X = data_train[:, 1:]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y = data_train[:, 0]

    X_test = data_test[:, 1:]
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    Y_test = data_test[:, 0]

    example_nn = Feed_Forward_NN(X, Y, X_test, Y_test)

    # example_nn.train()
    # example_nn.write_weights_to_h5()
    # example_nn.evaluate()

    example_nn.create_and_load_model('results/lstm_90lr_sp500_weights.h5')
    prediction = example_nn.predict()
    # prediction = le.inverse_transform(prediction)
    
    print(type(X_test))

    # print(X_test[0])
    plt.plot(prediction)
    plt.plot(Y_test)
    plt.title('SP500 Prediction vs Actual')
    plt.ylabel('Stock Price')
    plt.xlabel('Minutes')
    plt.legend(['Prediction', 'Actual'], loc='upper left')
    plt.show()

def lstm_01lr_sp500_model():
    data = pd.read_csv('data/numerical/sp500.csv')
    # data = pd.read_sql_query("select * from \"AAPL_ticker\"", con=engine)
    # print(type(data))
    data = data.drop(['DATE'], 1)
    
    n = data.shape[0] 

    # Scale data
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder as le
    scaler = MinMaxScaler(feature_range = (0, 1))
    data = data.values
    data = scaler.fit_transform(data)

    train_start = 0
    train_end = int(np.floor(0.8*n))

    test_start = train_end
    test_end = n

    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    X = data_train[:, 1:]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y = data_train[:, 0]

    X_test = data_test[:, 1:]
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    Y_test = data_test[:, 0]

    example_nn = Feed_Forward_NN(X, Y, X_test, Y_test)

    # example_nn.train()
    # example_nn.write_weights_to_h5()
    # example_nn.evaluate()

    # example_nn.write_history_to_file()
    example_nn.create_and_load_model('results/lstm_01lr_sp500_weights.h5')
    prediction = example_nn.predict()
    # prediction = le.inverse_transform(prediction)
    
    print(type(X_test))

    print(X_test[0])
    plt.plot(prediction)
    plt.plot(Y_test)
    plt.title('SP500 Prediction vs Actual')
    plt.ylabel('Stock Price')
    plt.xlabel('Minutes')
    plt.legend(['Prediction', 'Actual'], loc='upper left')
    plt.show()


# from tensorflow.python.client import device_lib

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

def curate_aapl_data():
    data = pd.read_csv('data/numerical/Apple-Twitter-Sentiment-DFE.csv', encoding='ISO-8859-1')
    print(data.columns.values)

    print('\n\n\n')
    data = data.drop(['_unit_state', 
                    'Unnamed: 12', 
                    'Unnamed: 13',
                    'sentiment_gold',
                    'query',
                    'id',
                    '_last_judgment_at',
                    'date'], 1)

    mask = data.sentiment == 'not_relevant'
    column_name = "sentiment"
    data.loc[mask, column_name] = 0

    data.to_csv('data/numerical/revamped_aapl.csv', sep=',', encoding='ISO-8859-1')


def lstm_01lr_aapl_sentiment():
    data = pd.read_csv('data/numerical/revamped_aapl.csv', encoding='ISO-8859-1')

    from keras.preprocessing.text import Tokenizer
    t = Tokenizer()

    tweets = data['text']
    data = data.drop(['text', '_unit_id', 'Unnamed: 0'], 1)
    t.fit_on_texts(tweets)
    # print(tweets)
    encoded_docs = t.texts_to_matrix(tweets, mode='count')
    print(encoded_docs)
    print(encoded_docs)
    # Convert from pd.df to numpy array

    # print(vectorizer.vocabulary_)
    
    # data = np.append(data, encoded_docs, 1)
    print(data)
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder as le
    scaler = MinMaxScaler(feature_range = (0, 1))
    data = data.values
    data = scaler.fit_transform(data)
    print(data)

    n = data.shape[0] 
    train_start = 0
    train_end = int(np.floor(0.8*n))

    test_start = train_end
    test_end = n

    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    X = data_train[:, 1:]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y = data_train[:, 0]

    X_test = data_test[:, 1:]
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    Y_test = data_test[:, 0]

    example_nn = LSTM_NN(X, Y, X_test, Y_test)
    example_nn.train()
    example_nn.write_weights_to_h5()
    example_nn.graph_accuracy()

    # example_nn.model = None
    example_nn.create_and_load_model('model.h5')
    prediction = example_nn.predict()

    print(type(X_test))

    print(X_test[0])
    plt.plot(prediction)
    plt.plot(Y_test)
    plt.title('SP500 Prediction vs Actual')
    plt.ylabel('Stock Price')
    plt.xlabel('Tweet #')
    plt.legend(['Prediction', 'Actual'], loc='upper left')
    plt.show()



def main():
    # lstm_sp500_model()
    # lstm_01lr_sp500_model()
    # get_available_gpus()
    lstm_01lr_aapl_sentiment()
    


if __name__ == "__main__":
    main()



