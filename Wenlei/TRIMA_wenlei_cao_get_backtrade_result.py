#!/usr/bin/env python
# coding: utf-8

# In[1]:


import backtrader as bt
import backtrader.indicators as btind
import pandas as pd
#import datetime
from datetime import datetime
from finta import TA
import matplotlib.pyplot as plt
import csv
import os

if os.path.exists("result.csv"):
    os.remove("result.csv")
else:
    print("The file does not exist")


# In[4]:


class TRIMA (bt.Strategy):
    # Moving average parameters
    params = (('pfast', 7), ('pslow', 17),)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        # Comment this line when running optimization
        print('%s, %s' % (dt.isoformat(), txt))
        return('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        # Order variable will contain ongoing order details/status
        self.order = None
        # Instantiate RSI
        #self.RSI = bt.indicators.RSI(self.datas[0], period=self.params.period)
        # fast moving average
        sma1 = bt.talib.TRIMA(self.datas[0], timeperiod=self.params.pfast)
        # slow moving average
        sma2 = bt.talib.TRIMA(self.datas[0], timeperiod=self.params.pslow)
        self.crossover = bt.indicators.CrossOver(sma1, sma2)
        self.filename = "result.csv"

        ''' Using the built-in crossover indicator
        self.crossover = bt.indicators.CrossOver(self.slow_sma, self.fast_sma)'''

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # An active Buy/Sell order has been submitted/accepted - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
                with open(self.filename, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    # csvwriter.writerow(self.fields)
                    row = [[self.log('BUY EXECUTED, %.2f' %
                                     order.executed.price)]]
                    csvwriter.writerows(row)
            elif order.issell():
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

        # Reset orders
        self.order = None

    def next(self):
        ''' Logic for using the built-in crossover indicator

        if self.crossover > 0: # Fast ma crosses above slow ma
            pass # Signal for buy order
        elif self.crossover < 0: # Fast ma crosses below slow ma
            pass # Signal for sell order
        '''

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # We are not in the market, look for a signal to OPEN trades

            # If the 20 SMA is above the 50 SMA
            if self.crossover > 0:
                #self.log('BUY CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()
            # Otherwise if the 20 SMA is below the 50 SMA
        elif self.crossover < 0:
            self.close()


# In[5]:


# use ML classification tuned param set for TRIMA, fast 7, slow 17
cerebro = bt.Cerebro()
cerebro.broker.setcash(30000.0)
# Set data parameters and add to Cerebro
data = bt.feeds.YahooFinanceCSVData(
    dataname="/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/Wenlei/VINIX_all.csv",
    fromdate=datetime(1990, 7, 31),
    todate=datetime(2020, 9, 23))
# settings for out-of-sample data
#fromdate=datetime.datetime(2018, 1, 1),
# todate=datetime.datetime(2019, 12, 25))

cerebro.adddata(data)

# Add strategy to Cerebro
cerebro.addstrategy(TRIMA)

# Default position size
cerebro.addsizer(bt.sizers.SizerFix, stake=100)

# if __name__ == '__main__':


def main():
    print('TRIMA param: pfast:%i, pslow:%i' %
          (TRIMA.params.pfast, TRIMA.params.pslow))
    # Run Cerebro Engine
    start_portfolio_value = cerebro.broker.getvalue()

    cerebro.run()

    end_portfolio_value = cerebro.broker.getvalue()
    pnl = end_portfolio_value - start_portfolio_value
    print('Starting Portfolio Value: %.2f' % start_portfolio_value)
    print('Final Portfolio Value: %.2f' % end_portfolio_value)
    print('PnL: %.2f' % pnl)
    # cerebro.plot()


# In[ ]:


# copy the output file to excel (no header, copy buy sell pair, if last one is only buy, then skip that one)
# save as csv, see the sample csv file for format
# the expectunity function will import the file as df and do calculation


# In[7]:
