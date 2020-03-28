DOTCSV = '.csv'
DOTTXT = '.txt'

DATA_DIR = './data/stocknet-dataset/'
POST_PROCESSING_DIR = './data/post_processing/'
RESULTS_DIR = './data/results/'
PRICE_PREPROCESSED_DIR = DATA_DIR + 'price/preprocessed/'
PRICE_RAW_DIR = DATA_DIR + 'price/raw/'
TWEET_PREPROCESSED_DIR = DATA_DIR + 'tweet/preprocessed/'
TWEET_RAW_DIR = DATA_DIR + 'tweet/raw/'
NN_RESULT_DIR = 'data/results/nn/'


POSITIVE_THRESHOLD = .5

EXPERIMENTS = {
    'sentiments_with_close': ['daily_pos_sent_avg', 
                                'daily_neg_sent_avg',
                                'Close',
                                'next_day_close'],
    'sentiments_with_open': ['daily_pos_sent_avg', 
                                'daily_neg_sent_avg',
                                'Open',
                                'next_day_close'],
    'sentiments_only': ['daily_pos_sent_avg', 
                        'daily_neg_sent_avg',
                        'next_day_close'],
}