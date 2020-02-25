from zipline.api import order_target, record, symbol
from talib import EMA
import matplotlib.pyplot as plt


def initialize(context):
    context.i = 0
    context.invested = False
    context.asset = symbol('AAPL')



def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 100:
        return

    trailing_window = data.history(context.asset, 'price', 40, '1d')
    if trailing_window.isnull().values.any():
        return

    # Compute averages
    # data.history() has to be called with the same params
    # from above and returns a pandas dataframe.
    thirty_day_mavg = data.history(context.asset, 'price', bar_count=30, frequency="1d").mean()
    hundred_day_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1d").mean()

    short_ema = EMA(trailing_window.values, timeperiod=20)
    long_ema = EMA(trailing_window.values, timeperiod=40)

    buy = False
    sell = False

    # Trading logic
    if thirty_day_mavg > hundred_day_mavg and not context.invested:
        # order_target orders as many shares as needed to
        # achieve the desired number of shares.
        buy = True
        order_target(context.asset, 100)
        context.invested = True


    elif thirty_day_mavg < hundred_day_mavg and context.invested:
        sell = True
        order_target(context.asset, 0)
        context.invested = False


    # Save values for later inspection
    record(AAPL=data.current(context.asset, 'price'),
            thirty_day_mavg=thirty_day_mavg,
            hundred_day_mavg=hundred_day_mavg,
            short_ema=short_ema[-1],
            long_ema=long_ema[-1],
            buy=buy,
            sell=sell)


def analyze(context, perf):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')

    print('Performance: ', perf.portfolio_value)

    ax2 = fig.add_subplot(212)

    perf['AAPL'].plot(ax=ax2)

    perf[[
        'thirty_day_mavg', 
        'hundred_day_mavg', ]].plot(ax=ax2) 
        # 'short_ema', 
        # 'long_ema'
        # ]].plot(ax=ax2)

    perf_trans = perf.ix[[t != [] for t in perf.transactions]]

    buys = perf_trans.ix[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
    sells = perf_trans.ix[[t[0]['amount'] < 0 for t in perf_trans.transactions]]

    ax2.plot(buys.index, perf.thirty_day_mavg.ix[buys.index],  '^', markersize=8, color='black')

    # for item in perf.thirty_day_mavg:
    #     print(item)

    ax2.plot(sells.index, perf.hundred_day_mavg.ix[sells.index], 'v', markersize=8, color='red')

    print('Do I sell: ', perf.thirty_day_mavg.ix[buys.index])


    # ax2.plot(buys.index, perf.short_ema.ix[buys.index], '>', markersize=8, color='orange')

    # ax2.plot(sells.index, perf.long_ema.ix[sells.index], '<', markersize=8, color='green')

    ax2.set_ylabel('price in $')
    plt.legend(loc=0)
    plt.show()