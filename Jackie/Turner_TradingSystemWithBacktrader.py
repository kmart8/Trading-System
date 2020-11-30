#!/usr/bin/env python
# coding: utf-8

# ### Model Estimation

# In[1]:


from matplotlib import warnings
import matplotlib
import numpy as np
import backtrader as bt
from datetime import datetime as dt
from backtrader.feeds import GenericCSVData
import backtrader.feeds as btfeeds
import datetime
import pandas as pd
import yfinance as yf
from finta import TA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
import matplotlib.pyplot as plt
# import to plot in Jupyter
#get_ipython().run_line_magic('matplotlib', 'inline')


def predict_close(stock_name, days_forward, start_date, train_end_date, end_date, plot_end_date):
    raw_data = yf.download(stock_name, start=start_date, end=end_date)

    data_with_indicators = raw_data.copy()
    # For ROC set the period to equal the number of days forward that we want to predict
    data_with_indicators['ROC'] = TA.ROC(
        data_with_indicators, period=days_forward)
    data_with_indicators['KAMA'] = TA.KAMA(data_with_indicators)
    data_with_indicators['RSI'] = TA.RSI(data_with_indicators)
    data_with_indicators['MOM'] = TA.MOM(data_with_indicators)

    indicator_name = 'Close_in_' + str(days_forward) + '_days'
    predicted_indicator_name = 'Predicted_Close'

    data_with_indicators[indicator_name] = data_with_indicators['Close'].shift(
        -days_forward)
    data_with_indicators = data_with_indicators.dropna()

    X_columns = ['ROC', 'KAMA', 'RSI', 'MOM']
    y_column = [indicator_name]
    model_data = data_with_indicators[start_date:
                                      train_end_date][X_columns + y_column]
    print(stock_name + ' indicator correlation:')
    # display(model_data.corr())

    X = model_data.loc[:, model_data.columns != y_column[0]]
    y = model_data.loc[:, model_data.columns == y_column[0]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=0)

    regr = RandomForestRegressor(max_depth=10, random_state=0)
    regr.fit(X_train, y_train)
    y_pred_train = regr.predict(X_train)
    y_pred = regr.predict(X_test)

    print(stock_name + ' model performance:')
    print('RMSE_train:', mean_squared_error(
        y_train, y_pred_train, squared=False))
    print('R2_train:', r2_score(y_train, y_pred_train))
    print('RMSE_test:', mean_squared_error(y_test, y_pred, squared=False))
    print('R2_test:', r2_score(y_test, y_pred))

    # Add prediction to dataset
    backtrader_data = data_with_indicators
    backtrader_data[predicted_indicator_name] = regr.predict(
        backtrader_data[X_columns])
    # Calc % price change based on price prediction
    backtrader_data['Predicted_Change_Pct'] = (
        backtrader_data[predicted_indicator_name]/backtrader_data[predicted_indicator_name].shift(days_forward)-1)

    # plot data with prediction
    # (backtrader_data[start_date:plot_end_date]
    # [[indicator_name, predicted_indicator_name]]).plot()
    # plt.title(stock_name + ' ' + predicted_indicator_name)
    # plt.show()

    # save csv file for import into backtrader
    filename = "/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/Jackie/backtrader_model_" + stock_name + '.csv'
    backtrader_data.to_csv(filename)


# In[19]:


# Set parameters, run model and generate input data here
days_forward = 3
start_date = '1990-01-01'
train_end_date = '2019-11-22'
end_date = '2020-11-22'
plot_end_date = '2020-11-22'


stocks = ['CRM', 'MSFT', 'CDNS']
for stock_name in stocks:
    predict_close(stock_name, days_forward, start_date,
                  train_end_date, end_date, plot_end_date)


# ### Backtrader Simulation

# In[20]:


# Formulas for expectunity calculation
# standard expectunity formula
def expectancy(wins, losses):

    try:
        AW = sum(wins)/len(wins)
        PW = len(wins)/(len(wins) + len(losses))

        AL = sum(losses)/len(losses)
        PL = len(losses)/(len(wins) + len(losses))

        return (AW * PW + AL * PL)/abs(AL)

    except ZeroDivisionError:
        return 0


def expectunity(wins, losses, strat_cal_days):

    try:
        # Calculate the opportunity value
        num_trades = len(wins) + len(losses)
        strat_cal_days = max(strat_cal_days, 1)  # avoid divide by zero
        opportunities = num_trades * 365/strat_cal_days

        return expectancy(wins, losses) * opportunities

    except ZeroDivisionError:
        return 0


# In[21]:


class GenericCSV_Indicators(GenericCSVData):
    # Add a predicted line to the inherited ones from the base class
    lines = ('Predicted_Change_Pct',)
    # add the parameter to the parameters inherited from the base class
    params = (('Predicted_Change_Pct', 13),)


def get_data_with_indicators(filename):
    data = GenericCSV_Indicators(
        dataname=filename,
        # simulation period
        fromdate=dt(2016, 12, 31),
        todate=dt(2019, 12, 31),
        nullvalue=0.0,
        dtformat=('%Y-%m-%d'),
        datetime=0,
        high=2,
        low=3,
        open=1,
        close=4,
        volume=5,
        openinterest=-1,
        # extra indicators
        Predicted_Change_Pct=13

    )
    return data


# In[22]:


class ModelMultiStrategy(bt.Strategy):

    #params = (('pfast',20),('pslow',50),)
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        # Order variable will contain ongoing order details/status
        self.order = None
        self.inmarket = False
        self.inmarketdataindex = None
        # Wins/Losses, prices and trading days for expectunity
        self.wins = []
        self.losses = []
        self.firsttradedate = None
        self.buyprice = 0.00
        self.sellprice = 0.00
        self.size = 0.00
        self.expect = 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Active Buy/Sell order submitted/accepted - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                # track first trade date on first buy
                if self.firsttradedate == None:
                    self.firsttradedate = self.datas[0].datetime.date(0)
                self.buyprice = order.executed.price
                self.inmarket = True
                self.size = order.size
                self.log('trade size: %.2f' % self.size)
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.sellprice = order.executed.price
                self.inmarket = False
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

                ##########################################
                # EXPECTUNITY:
                # check execution prices to calculate win or loss
                winloss = (self.sellprice - self.buyprice)*self.size

                if winloss > 0:
                    self.wins.append(winloss)
                else:
                    self.losses.append(winloss)

                currentdate = self.datas[0].datetime.date(0)
                tradingdays = (currentdate - self.firsttradedate).days
                expect = expectunity(self.wins, self.losses, tradingdays)
                self.log('trade pnl %.2f, winnings %.2f, #wins %d, losses %.2f, #losses %d, calendar days %d, expectunity %.2f' %
                         (winloss, sum(self.wins), len(self.wins), sum(self.losses), len(self.losses), tradingdays, expect))
                ##########################################

            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        # Reset orders
        self.order = None

    def next(self):
        if self.order:
            return

        # Check if we are in the market:
        # We are not in the market, look for a signal to OPEN trades
        if self.inmarket == False:
            pred_changes = [self.datas[0].Predicted_Change_Pct[0],
                            self.datas[1].Predicted_Change_Pct[0], self.datas[2].Predicted_Change_Pct[0]]

            if max(pred_changes) > 0:
                self.inmarketdataindex = np.argmax(pred_changes)
                d = self.datas[self.inmarketdataindex]

                self.log('BUY CREATE, %.2f' % d.close[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy(data=d)

                return

        # We are already in the market, look for a signal to CLOSE trades
        elif len(self) >= (self.bar_executed + 3):
            self.log('CLOSE CREATE, %.2f' %
                     self.datas[self.inmarketdataindex].close[0])
            self.order = self.sell(data=self.datas[self.inmarketdataindex])


# In[23]:

# Run Simulation:

# fix stake
stake = 10

# import files for each stock
datalist = [
    ('backtrader_model_CRM.csv', 'CRM'),
    ('backtrader_model_MSFT.csv', 'MSFT'),
    ('backtrader_model_CDNS.csv', 'CDNS'),
]

cerebro = bt.Cerebro()
for i in range(len(datalist)):
    data = get_data_with_indicators(filename=datalist[i][0])
    cerebro.adddata(data, name=datalist[i][1])

# Add strategy to Cerebro
cerebro.addstrategy(ModelMultiStrategy)

# Default position size
cerebro.addsizer(bt.sizers.SizerFix, stake=stake)
#cerebro.addsizer(bt.sizers.PercentSizer, percents=5)

# if __name__ == '__main__':


def jackie():
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


# In[ ]:
