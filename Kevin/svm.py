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
