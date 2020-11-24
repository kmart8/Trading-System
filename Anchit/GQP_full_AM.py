#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.exceptions import DataConversionWarning
import warnings 

warnings.filterwarnings(action='ignore',category=DataConversionWarning)


# In[2]:


#import data
stock1 = yf.Ticker("ABBV")
data1 = stock1.history(period='2y',interval='60m')
abbv = pd.DataFrame(data1) 

stock2 = yf.Ticker('BMY')
data2 = stock2.history(period='2y',interval='60m')
bmy = pd.DataFrame(data2)


# In[3]:


#expectunity functions
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
        #strat_cal_days = len(trans)
        #strat_cal_days= max(strat_cal_days,1)  # avoid divide by zero
        opportunities = num_trades * 365/strat_cal_days
    
        return expectancy(wins, losses) * opportunities
    
    except ZeroDivisionError:
        return 0


# In[4]:


###Insert Technical Indicators###

#Rolling Mean
def rolling_mean(df,input):
    x = [2,3,4,5,6]
    for i in x:
        rm = df[[input]].rolling(i).mean()
        df[f'rm{i}_{input}'] = rm

#Exponential Moving Average
def exponential_moving(df,input):
    x = [2,3,4,5,6]
    for i in x:
        ema = df[[input]].ewm(span=i, adjust=False).mean()
        df[f'ema{i}_{input}'] = ema

#Bollinger Bands
def bollinger(df,input):
    x = [2,3,4,5,6]
    for i in x:
        rm = df[[input]].rolling(i).mean()
        upper = rm + 2*df[[input]].rolling(i).std()
        lower = rm - 2*df[[input]].rolling(i).std()
        df[f'boll_upper{i}_{input}'] = upper
        df[f'boll_lower{i}_{input}'] = lower


# In[5]:


### Driver for Technical Functions ###
func = [rolling_mean,exponential_moving,bollinger]
data = [abbv,bmy]
column = ['Open','High','Low','Close']

def technical(func,data,column):
    for i in func:
        for j in data:
            for k in column:
                i(j,k)

technical(func,data,column)


# In[6]:


### Prepare for Machine Learning ###
#split data into train and test
abbv_train = abbv[7:3000]
abbv_test = abbv[3001:]
bmy_train = bmy[7:3000]
bmy_test = bmy[3001:]


# In[7]:


#Define Models
attribute = []
score = []

def abbv_model(input):
    if f'abbv_svm_pred_{input}' not in attribute:
        svm_model = svm.SVR()
        svm_model.fit(abbv_train[[input]],abbv_train[['Close']])
        svm_pred = svm_model.predict(abbv_test[[input]])
        svm_pred = list(svm_pred)
        result = svm_model.score(abbv_test[[input]],abbv_test[['Close']])
        #print(f'Abbvie score using {input} is {result}')
        attribute.append(f'abbv_svm_pred_{input}')
        score.append(round(result,4))
        x = [0] * 3001
        z = x + svm_pred
        abbv[f'svm_pred_{input}'] = z

def bmy_model(input):
    if f'bmy_svm_pred_{input}' not in attribute:
        svm_model = svm.SVR()
        svm_model.fit(bmy_train[[input]],bmy_train[['Close']])
        svm_pred = svm_model.predict(bmy_test[[input]])
        svm_pred = list(svm_pred)
        result = svm_model.score(bmy_test[[input]],bmy_test[['Close']])
        #print(f'BMS score using {input} is {result}')
        attribute.append(f'bmy_svm_pred_{input}')
        score.append(round(result,4))
        x = [0] * 3001
        z = x + svm_pred
        bmy[f'svm_pred_{input}'] = z


# In[8]:


#Run Models
def model(column):
    for i in [abbv_model,bmy_model]:
        for j in column:
            if 'Close' not in j:
                i(j)

model(list(abbv.columns)+list(bmy.columns))

result_df = pd.DataFrame(data=list(zip(attribute,score)),columns=['attribute','score'])
result_df.sort_values(by='score',ascending=False)
result_df.to_excel('result.xlsx')


# In[9]:


#From the results excel file generated I will pick the best performing models
#and technical indicators and proceed with that

'''
BMY:
Low
High
ema2_Low

ABBV:
EMA6_High
High
EMA5_High
'''


# In[10]:


#Define buy/sell criteria
'''
BUY: if the predicted value is greater than current then buy
SELL: if the predicted value is decreasing for next point then sell OR if position is
        held for 4 hours OR if youve made 10% profit exit
'''

bmy_low = bmy[['Close','svm_pred_Low']][3001:]
bmy_high = bmy[['Close','svm_pred_High']][3001:]
bmy_ema2_low = bmy[['Close','svm_pred_ema2_Low']][3001:]

abbv_ema6_high = abbv[['Close','svm_pred_ema6_High']][3001:]
abbv_high = abbv[['Close','svm_pred_High']][3001:]
abbv_ema5_high = abbv[['Close','svm_pred_ema5_High']][3001:]

all = [bmy_low,bmy_high,bmy_ema2_low,abbv_ema6_high,abbv_high,abbv_ema5_high]
for i in all:
    i['c'] = np.arange(len(i))
    i.set_index('c', inplace=True)


# In[11]:


'''
#Define current point in time
current_bmy = bmy['Close'][3000]
current_abbv = abbv['Close'][3000]

#Define current position in/out of market
in_market = False
bought_at = 0
counter = 0

#count number of trades and wins and loses <CALCULATED AT TIME OF SELL>
trade = 0
win = 0
lose = 0

#store number of trades and associated values along with exit conditions
lst_trade = []
lst_win = []
lst_lose = []
lst_cash = []
lst_profit = []
lst_exit_ml = []
lst_exit_counter = []

#cash stack 
cash = 100000
cash_invest = 0
num_stocks = 0

#See how well this works
for i in bmy_low.index:
    x = bmy_low['svm_pred_Low'][i+1]
    y = bmy_low['Close'][i]
    
    #Look to jump in the market
    if in_market == False:
        if current_bmy > x:
            #print(f'WAITING OOM -- {current_bmy} which is greater than {x}. Reassign and move on.')
            current_bmy = bmy_low['Close'][i+1]
            continue
            
        if current_bmy <= x:
            #print(f'INCREASE PREDICTED -- Current is {current_bmy} which is less than {x}')
            in_market = True
            bought_at = current_bmy
            cash_invest = cash*0.2
            num_stocks = cash_invest/bought_at
            continue
            
    if in_market == True:
        
        #hold if market increasing, increase hold counter +1
        if bought_at <= x:
            #print(f'WAITING IN -- bought at {bought_at}. Holding due to comparison vs {x}')
            counter += 1
        
        #exit market if hold counter >= 4 OR decrease predicted
        if bought_at > x or counter >= 4:
            
            #decrease in market
            if bought_at > x:
                #print(f'DECREASE PREDICTED --  Selling at {y}. Bought at {bought_at}')
                lst_exit_ml.append(1)
                lst_exit_counter.append(0)
            
            #hold counter >= 4
            if counter >= 4:
                #print(f'COUNTER EXCEEDED - selling at {y}. Bought at {bought_at}')
                lst_exit_ml.append(0)
                lst_exit_counter.append(1)
            
            #exit market calculations
            trade += 1
            lst_trade.append(1)
            
            if y - bought_at > 0:
                win += 1
                lst_win.append(win)
                lst_lose.append(lose)
                
            if y - bought_at <= 0:
                lose += 1
                lst_win.append(win)
                lst_lose.append(lose)
                
            profit = (num_stocks * y) - cash_invest
            lst_profit.append(profit)
            cash += profit
            lst_cash.append(cash)
            
            #output trades/wins/loses
            print(f'Trades = {trade}. Wins = {win}. Loses = {lose}. Cash = {round(cash,1)}. Profit = {round(profit,1)}.')
            
            #reset counters and market position
            in_market = False
            bought_at = 0
            counter = 0
            cash_invest = 0
            num_stocks = 0
            current_bmy = y
            exit_ml = 0
            exit_counter = 0
            
'''            


# In[19]:


def bmy_trader(model,column):
    #Define current point in time
    current_bmy = bmy['Close'][3000]
    #current_abbv = abbv['Close'][3000]
    
    #Define current position in/out of market
    in_market = False
    bought_at = 0
    counter = 0

    #count number of trades and wins and loses <CALCULATED AT TIME OF SELL>
    trade = 0
    win = 0
    lose = 0

    #store number of trades and associated values along with exit conditions
    lst_trade = []
    lst_win = []
    lst_lose = []
    lst_cash = []
    lst_profit = []
    lst_exit_ml = []
    lst_exit_counter = []
    
    e_win = []
    e_lose = []

    #cash stack 
    cash = 100000
    cash_invest = 0
    num_stocks = 0
    
    for i in model[:-1].index:
        x = model[column][i+1]
        y = model['Close'][i]
    
        #Look to jump in the market
        if in_market == False:
            if current_bmy > x:
                #print(f'WAITING OOM -- {current_bmy} which is greater than {x}. Reassign and move on.')
                current_bmy = model['Close'][i+1]
                continue

            if current_bmy <= x:
                #print(f'INCREASE PREDICTED -- Current is {current_bmy} which is less than {x}')
                in_market = True
                bought_at = current_bmy
                cash_invest = cash*0.05
                num_stocks = cash_invest/bought_at
                continue

        if in_market == True:

            #hold if market increasing, increase hold counter +1
            if bought_at <= x:
                #print(f'WAITING IN -- bought at {bought_at}. Holding due to comparison vs {x}')
                counter += 1

            #exit market if hold counter >= 4 OR decrease predicted
            if bought_at > x or counter >= 4:

                #decrease in market
                if bought_at > x:
                    #print(f'DECREASE PREDICTED --  Selling at {y}. Bought at {bought_at}')
                    lst_exit_ml.append(1)
                    lst_exit_counter.append(0)

                #hold counter >= 4
                if counter >= 4:
                    #print(f'COUNTER EXCEEDED - selling at {y}. Bought at {bought_at}')
                    lst_exit_ml.append(0)
                    lst_exit_counter.append(1)

                #exit market calculations
                trade += 1
                lst_trade.append(trade)

                if y - bought_at > 0:
                    win += 1
                    lst_win.append(win)
                    lst_lose.append(lose)

                if y - bought_at <= 0:
                    lose += 1
                    lst_win.append(win)
                    lst_lose.append(lose)

                profit = (num_stocks * y) - cash_invest
                lst_profit.append(profit)
                cash += profit
                lst_cash.append(cash)
                
                #append to expectunity calculations
                if profit > 0:
                    e_win.append(profit)
                if profit < 0:
                    e_lose.append(profit)

                #output trades/wins/loses
                #print(f'Trades = {trade}. Wins = {win}. Loses = {lose}. Cash = {round(cash,1)}. Profit = {round(profit,1)}.')

                #reset counters and market position
                in_market = False
                bought_at = 0
                counter = 0
                cash_invest = 0
                num_stocks = 0
                current_bmy = y
                exit_ml = 0
                exit_counter = 0
                
    #create output dataframe from the trader
    df = pd.DataFrame(list(zip(lst_trade,lst_win,lst_lose,lst_cash,lst_profit,lst_exit_ml,lst_exit_counter)),
                     columns=['trade#','win_count','lose_count','cash','profit','exit_ML','exit_counter'])
    
    df.to_excel(f'BMY_output_{column}.xlsx')
    
    e = expectunity(e_win,e_lose,57600)
    print(f'Expectunity for BMY using {column} is = {round(e,3)} with a total value of {round(cash,1)} in {trade} trades.')


# In[20]:


def abbv_trader(model,column):
    #Define current point in time
    #current_bmy = bmy['Close'][3000]
    current_abbv = abbv['Close'][3000]
    
    #Define current position in/out of market
    in_market = False
    bought_at = 0
    counter = 0

    #count number of trades and wins and loses <CALCULATED AT TIME OF SELL>
    trade = 0
    win = 0
    lose = 0

    #store number of trades and associated values along with exit conditions
    lst_trade = []
    lst_win = []
    lst_lose = []
    lst_cash = []
    lst_profit = []
    lst_exit_ml = []
    lst_exit_counter = []
    
    e_win = []
    e_lose = []

    #cash stack 
    cash = 100000
    cash_invest = 0
    num_stocks = 0
    
    for i in model[:-1].index:
        x = model[column][i+1]
        y = model['Close'][i]
    
        #Look to jump in the market
        if in_market == False:
            if current_abbv > x:
                #print(f'WAITING OOM -- {current_bmy} which is greater than {x}. Reassign and move on.')
                current_abbv = model['Close'][i+1]
                continue

            if current_abbv <= x:
                #print(f'INCREASE PREDICTED -- Current is {current_bmy} which is less than {x}')
                in_market = True
                bought_at = current_abbv
                cash_invest = cash*0.05
                num_stocks = cash_invest/bought_at
                continue

        if in_market == True:

            #hold if market increasing, increase hold counter +1
            if bought_at <= x:
                #print(f'WAITING IN -- bought at {bought_at}. Holding due to comparison vs {x}')
                counter += 1

            #exit market if hold counter >= 4 OR decrease predicted
            if bought_at > x or counter >= 4:

                #decrease in market
                if bought_at > x:
                    #print(f'DECREASE PREDICTED --  Selling at {y}. Bought at {bought_at}')
                    lst_exit_ml.append(1)
                    lst_exit_counter.append(0)

                #hold counter >= 4
                if counter >= 4:
                    #print(f'COUNTER EXCEEDED - selling at {y}. Bought at {bought_at}')
                    lst_exit_ml.append(0)
                    lst_exit_counter.append(1)

                #exit market calculations
                trade += 1
                lst_trade.append(trade)

                if y - bought_at > 0:
                    win += 1
                    lst_win.append(win)
                    lst_lose.append(lose)

                if y - bought_at <= 0:
                    lose += 1
                    lst_win.append(win)
                    lst_lose.append(lose)

                profit = (num_stocks * y) - cash_invest
                lst_profit.append(profit)
                cash += profit
                lst_cash.append(cash)

                #output trades/wins/loses
                #print(f'Trades = {trade}. Wins = {win}. Loses = {lose}. Cash = {round(cash,1)}. Profit = {round(profit,1)}.')

                #reset counters and market position
                in_market = False
                bought_at = 0
                counter = 0
                cash_invest = 0
                num_stocks = 0
                current_bmy = y
                exit_ml = 0
                exit_counter = 0
                
                #append to expectunity calculations
                if profit > 0:
                    e_win.append(profit)
                if profit < 0:
                    e_lose.append(profit)
                
    #create output dataframe from the trader
    df = pd.DataFrame(list(zip(lst_trade,lst_win,lst_lose,lst_cash,lst_profit,lst_exit_ml,lst_exit_counter)),
                     columns=['trade#','win_count','lose_count','cash','profit','exit_ML','exit_counter'])
    
    df.to_excel(f'ABBV_output_{column}.xlsx')
    
    e = expectunity(lst_win,lst_lose,57600)
    print(f'Expectunity for ABBV using {column} is = {round(e,3)} with a total value of {round(cash,1)} in {trade} trades.')


# In[21]:


bmy_trader(bmy_low,'svm_pred_Low')
bmy_trader(bmy_high,'svm_pred_High')
bmy_trader(bmy_ema2_low,'svm_pred_ema2_Low')

abbv_trader(abbv_ema6_high,'svm_pred_ema6_High')
abbv_trader(abbv_high,'svm_pred_High')
abbv_trader(abbv_ema5_high,'svm_pred_ema5_High')


# In[ ]:




