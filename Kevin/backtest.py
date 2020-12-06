import backtrader as bt
import base64
from io import BytesIO
import csv
# from svm import *
import os
import numpy as np

if os.path.exists("result.csv"):
    os.remove("result.csv")
else:
    print("The file does not exist")

from sklearn import svm
import pandas as pd
import numpy as np
from finta import TA

# Add Data
import yfinance as yf
stock = yf.Ticker("AMD")
p = 0.85
df = stock.history(period="max", interval="1d")
df = df.rename(columns={"Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume"})
df = df.drop(columns=["Dividends", "Stock Splits"])
df["VBM"] = TA.VBM(df, roc_period=6, atr_period=2)
df = df[df.volume != None]
df = df[(df.VBM <= 100000000000000000000000) &
        (df.VBM >= -100000000000000000000000)]

l = []
for i in df.iterrows():
    if i[1]["close"] - i[1]["open"] > 0:
        l.append(1)
    elif i[1]["close"] - i[1]["open"] < 0:
        l.append(0)
    else:
        l.append(0)
df["Direction"] = l

train = df[0:round(p*df.shape[0])].drop(columns=["Direction",
                                                 "close", "low", "high", "volume"])
test = df[round(p*df.shape[0]):df.shape[0]
          ].drop(columns=["Direction", "close", "low", "high", "volume"])

trainDirection = df["Direction"][0:round(p*df.shape[0])]
testDirection = df["Direction"][round(p*df.shape[0]):df.shape[0]]

clf = svm.SVC()
clf = clf.fit(train, trainDirection)


class SVMStrategy(bt.Strategy):

    params = (
        ('stake', 100),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
        return('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.sizer.setsizing(self.params.stake)

        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose1 = self.datas[0].close
        self.high = self.datas[0].high
        self.chan_exit_l = bt.ind.Highest(
            self.high, period=22) - 3 * bt.ind.ATR(period=22)
        self.rsa = bt.ind.RSI_SMA(period=14)
        self.order = None
        self.filename = "result.csv"
        self.W = []
        self.L = []

    def next(self):

        # check for open orders
        if self.order:
            return

        # check if we are in the market
        if not self.position:

            # We are not in the market, look for a signal to OPEN trades
            if clf.predict(np.array([self.dataclose1[0], self.rsa[0]]).reshape(1, -1)) == 1 and \
                    self.dataclose1[0] < self.chan_exit_l[0] and \
                    self.dataclose1[-1] >= self.chan_exit_l[-1] and \
                    self.rsa <= 35:

                # self.log('BUY CREATE, price=%.2f  %.2f' % (
                #     self.dataclose1[0],
                #     # self.chan_exit_l[0],
                #     self.rsa[0],
                # ))
                global temp
                temp = self.dataclose1[0]
                self.buy()
        else:
            # We are already in the market, look for a signal to CLOSE trades
            # if len(self) >= (self.bar_executed + 5):
            #     self.log(f'CLOSE CREATE {self.dataclose1[0]:2f}')
            #     self.order = self.close()
            if self.dataclose1[0] >= temp * 1.03:
                # self.log(f'CLOSE CREATE {self.dataclose1[0]:2f}')
                self.order = self.close()

            elif self.dataclose1[0] <= temp * 0.985:
                # self.log(f'CLOSE CREATE {self.dataclose1[0]:2f}')
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
                with open(self.filename, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    # csvwriter.writerow(self.fields)
                    row = [[self.log('BUY EXECUTED, %.2f' %
                                     order.executed.price)]]
                    csvwriter.writerows(row)
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
                with open(self.filename, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    # csvwriter.writerow(self.fields)
                    row = [[self.log('SELL EXECUTED, %.2f' %
                                     order.executed.price)]]
                    csvwriter.writerows(row)
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        if trade.pnl > 0:
            self.W.append(trade.pnl)
        elif trade.pnl <= 0:
            self.L.append(trade.pnl)

        # expectun = expectunity(self.W, self.L, 500)
        # self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f EXPECTUNITY %.2f WINS %d LOSSES %d' %
        #          (trade.pnl, trade.pnlcomm, expectun, len(self.W), len(self.L)))


def backtest():

    cerebro = bt.Cerebro()

    cerebro.broker.setcash(30000.0)
    cerebro.addstrategy(SVMStrategy)

    # Add Data
    import yfinance as yf
    stock = yf.Ticker("AMD")

    # Clean Data
    data = stock.history(start="2019-02-19", end="2020-02-19", interval="1h")
    data = data.drop(columns=["Dividends", "Stock Splits"])
    data = data[data.Volume != 0]

    # Put Data into Cerebro
    data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data)

    startValue = cerebro.broker.getvalue()
    print('Starting Portfolio Value: %.2f' % startValue)

    cerebro.run()

    endValue = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % endValue)

    perc = (endValue - startValue) / startValue * 100

    # Print percentage change in the value
    print("pct_chg=%.2f%%" % (perc))
    # cerebro.plot()


# if __name__ == "__main__":
#     backtest()
