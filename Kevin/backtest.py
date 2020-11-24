import backtrader as bt
from strategies import *
from expectunity import *


def backtest():

    #     # Read CSV (From Yahoo) to Pandas dataframe
    #     df1 = read_csv("%s.HK.csv" % sid)

    #     # Some Yahoo dataframe had zero volume, exclude them
    #     df1 = df1[df1.Volume != 0]

    # Back test the BuyChandelierStrategy with BackTrader
    cerebro = bt.Cerebro()

    cerebro.broker.setcash(1000000.0)
#     cerebro.broker.setcommission(commission=0.0035)
    cerebro.addstrategy(SVMStrategy)

    # Add Data
    import yfinance as yf
    stock = yf.Ticker("AMD")

    # Clean Data
    data = stock.history(start="2018-12-19", end="2020-02-06", interval="1h")
    data = data.drop(columns=["Dividends", "Stock Splits"])
    data = data[data.Volume != 0]

    # Put Data into Cerebro
    data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data)

#     data = bt.feeds.PandasData(dataname=df1)
#     cerebro.adddata(data)

    startValue = cerebro.broker.getvalue()
    print('Starting Portfolio Value: %.2f' % startValue)

    cerebro.run()

    endValue = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % endValue)

    perc = (endValue - startValue) / startValue * 100

    # Print percentage change in the value
    print("pct_chg=%.2f%%" % (perc))
    cerebro.plot()

    #expect = expectancy(SVMStrategy.W, SVMStrategy.L)
    #expectun = expectunity()
    #print("Expectunity", expect)


def main():
    # Backtest a list of HKEX stocks.
    #     sidList = ["0027", "0066", "0762", "1038", "1928", "2318", "2388"]

    #     for sid in sidList:
    backtest()
