import numpy as np
from svm import *
import backtrader as bt
from expectunity import *

# Create a Stratey
# will the losses be negative or positive


class SVMStrategy(bt.Strategy):

    params = (
        ('stake', 1000),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.sizer.setsizing(self.params.stake)

        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose1 = self.datas[0].close
        self.high = self.datas[0].high
        self.chan_exit_l = bt.ind.Highest(
            self.high, period=22) - 3 * bt.ind.ATR(period=22)
        self.rsa = bt.ind.RSI_SMA(period=14)
        self.order = None
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

                self.log('BUY CREATE, price=%.2f  %.2f' % (
                    self.dataclose1[0],
                    # self.chan_exit_l[0],
                    self.rsa[0],
                ))
                global temp
                temp = self.dataclose1[0]
                self.buy()
        else:
            # We are already in the market, look for a signal to CLOSE trades
            # if len(self) >= (self.bar_executed + 5):
            #     self.log(f'CLOSE CREATE {self.dataclose1[0]:2f}')
            #     self.order = self.close()
            if self.dataclose1[0] >= temp * 1.03:
                self.log(f'CLOSE CREATE {self.dataclose1[0]:2f}')
                self.order = self.close()

            elif self.dataclose1[0] <= temp * 0.985:
                self.log(f'CLOSE CREATE {self.dataclose1[0]:2f}')
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

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

        expectun = expectunity(self.W, self.L, 500)
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f EXPECTUNITY %.2f WINS %d LOSSES %d' %
                 (trade.pnl, trade.pnlcomm, expectun, len(self.W), len(self.L)))
