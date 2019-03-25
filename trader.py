

from zipline.api import order_target, record, symbol
import matplotlib.pyplot as plt

def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')


def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 300:
        return

    # Compute averages
    # data.history() has to be called with the same params
    # from above and returns a pandas dataframe.
    thirty_day_mavg = data.history(context.asset, 'price', bar_count=30, frequency="1d").mean()
    hundred_day_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1d").mean()

    print(thirty_day_mavg)

    # Trading logic
    if thirty_day_mavg > hundred_day_mavg:
        # order_target orders as many shares as needed to
        # achieve the desired number of shares.
        order_target(context.asset, 100)
    elif thirty_day_mavg < hundred_day_mavg:
        order_target(context.asset, 0)

    # Save values for later inspection
    record(AAPL=data.current(context.asset, 'price'),
           thirty_day_mavg=thirty_day_mavg,
           hundred_day_mavg=hundred_day_mavg)


def analyze(context, perf):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')

    ax2 = fig.add_subplot(212)
    # print(perf)
    perf['AAPL'].plot(ax=ax2)
    perf[['thirty_day_mavg', 'hundred_day_mavg']].plot(ax=ax2)

    perf_trans = perf.ix[[t != [] for t in perf.transactions]]
    buys = perf_trans.ix[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
    sells = perf_trans.ix[
        [t[0]['amount'] < 0 for t in perf_trans.transactions]]

    ax2.plot(buys.index, perf.thirty_day_mavg.ix[buys.index],
             '^', markersize=15, color='black')

    ax2.plot(sells.index, perf.thirty_day_mavg.ix[sells.index],
             'v', markersize=15, color='red')

    ax2.set_ylabel('price in $')
    plt.legend(loc=0)
    plt.show()