# please change the file path to yours
import pandas as pd
from datetime import datetime

# create function to convert export csv to df


def convert_outputcsv_toDF(path):
    # r'C:\Users\Wenlei\Desktop\GQP\bt_output_regression.csv'
    df = pd.read_csv(path, header=None)
    df = pd.concat([df[0].str.split(', ', expand=True)], axis=1)
    df1 = df[df[1] == 'BUY EXECUTED']
    df1 = df1.reset_index(drop=True)
    df2 = df[df[1] == 'SELL EXECUTED']
    df2 = df2.reset_index(drop=True)
    df = pd.merge(df1[[0, 2]], df2[[0, 2]], left_index=True, right_index=True)
    df.columns = ['Entry date', 'Entry price', 'Exit date', 'Exit price']
    df[['Entry price', 'Exit price']] = df[[
        'Entry price', 'Exit price']].astype(float)
    return df

# take convert df and calculate expectunity, notice this function only handle long sale not short sale


def calculateExpectunity(df,  NumnberofShare):
   # df.columns = ['Entry date', 'Entry price', 'Exit date', 'Exit price']
    NumnberofShare = NumnberofShare
    df['Profit/Loss'] = (df['Exit price'] - df['Entry price'])*NumnberofShare
    avergeloss = df.loc[df['Profit/Loss'] < 0, ['Profit/Loss']
                        ].sum()/df.loc[df['Profit/Loss'] < 0, ['Profit/Loss']].count()
    df['Average Loss'] = pd.Series([abs(avergeloss)
                                    for x in range(len(df.index))])
    df['Largest Loss'] = pd.Series(
        [abs(df.loc[df['Profit/Loss'] < 0, ['Profit/Loss']].min()) for x in range(len(df.index))])
    df['R Mult 1'] = df['Profit/Loss']/df['Average Loss']
    df['R Mult 2'] = df['Profit/Loss']/df['Largest Loss']
    firsttradeday = df['Entry date'].min()
    lasttradeday = df['Entry date'].max()
    #print(firsttradeday, lasttradeday )
    firsttradeday = datetime.strptime(firsttradeday, '%Y-%m-%d').date()
    lasttradeday = datetime.strptime(lasttradeday, '%Y-%m-%d').date()
    strategycalendarday = (lasttradeday - firsttradeday).days
    #print (strategycalendarday)
    numberoftrade = df['Entry date'].count()
    Expectancy = df['R Mult 1'].sum() / numberoftrade
    # Expectancy
    opportunity = numberoftrade*365 / strategycalendarday
    # opportunity
    Expectunity = Expectancy*opportunity
    Expectunity = float(Expectunity.to_string().replace('Profit/Loss   ', ''))
    return Expectunity


def combined_expectunity_calcuation(path, NumnberofShare):
    df = convert_outputcsv_toDF(path)
    expectunity = calculateExpectunity(df,  NumnberofShare)
    return expectunity


# In[9]:


# calculate expectunity
# print((combined_expectunity_calcuation(
#     "/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/Kevin/result.csv", 1000)))


# In[ ]:
