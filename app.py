from typing import final
from flask import Flask, render_template, Markup
from flask_table import Table, Col
import os

# Kevin's files
from Kevin import *
from Kevin.backtest import *

# Wenlei's files
from Wenlei.TRIMA_wenlei_cao_get_backtrade_result import *

# Jackie's files
from Jackie.Turner_TradingSystemWithBacktrader import *

# Jordan's files

# Anchit's files

# Expectunity Calculation
from expectunityCalc import *


class ResultsTable(Table):
    stock = Col("Stock")
    initialValue = Col("Initial Value")
    finalValue = Col("Final Value")
    PnL = Col("Profit / Loss")
    pct = Col("Percentage Change")
    ex = Col("Expectunity")


class Result(object):
    def __init__(self, stock, initialValue, finalValue, PnL, pct, ex):
        self.stock = stock
        self.initialValue = initialValue
        self.finalValue = finalValue
        self.PnL = PnL
        self.pct = pct
        self.ex = ex


class PortfolioTable(Table):
    stock = Col("Stock")
    initialValue = Col("Initial Value")
    finalValue = Col("Final Value")
    PnL = Col("Profit / Loss")
    pct = Col("Percentage Change")
    avgwins = Col("Avg Wins")
    avglosses = Col("Avg Losses")
    wins = Col("Wins")
    losses = Col("Losses")
    totaltrades = Col("Total Trades")
    ex = Col("Expectunity")


class Portfolio(object):
    def __init__(self, stock, initialValue, finalValue, PnL, pct, avgwins, avglosses, wins, losses, totaltrades, ex):
        self.stock = stock
        self.initialValue = initialValue
        self.finalValue = finalValue
        self.PnL = PnL
        self.pct = pct
        self.avgwins = avgwins
        self.avglosses = avglosses
        self.wins = wins
        self.losses = losses
        self.totaltrades = totaltrades
        self.ex = ex


app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template("index.html")


@app.route("/results")
def results():
    if os.path.exists("result.csv"):
        os.remove("result.csv")
    else:
        print("The file does not exist")
    r = []
    backtest()
    expectunityKevin = combined_expectunity_calcuation(
        "/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/result.csv", 1000)
    r.append(Result("Electronics", 10000, 11136,
                    1136, 0.0378, 10.05))

    main()
    expectunityWenlei = combined_expectunity_calcuation(
        "/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/result.csv", 100)
    r.append(Result("Mutual Funds", 10000, 10986,
                    986, 986/10000, 3.85))
    jackie()
    r.append(Result("ESG Stocks", 10000, 11483.60, 1483.60, 0.14836, 17.81))
    r.append(Result("BioMed Stocks", 10000, 11066.16,
                    1066.16, 1066.16/10000, 13.55))
    r.append(Result("Raytheon", 10000, 13125, 3125, 3125/10000, 13.008))
    p = []
    p.append(Portfolio("Total Portfolio", 50000, 13125+11066.16+11483+10986 +
                       11136, ((13125+11066.16+11483+10986+11136)-50000).__round__, 0.15592, 22.93, 12.18, 511, 319, 511+319, (10.5 + 3.85 + 13.55 + 13.008 + 17.81)/5))
    resultsTable = ResultsTable(r)
    portfolioTable = PortfolioTable(p)
    return render_template("results.html", expectunityKevin=Markup(expectunityKevin),
                           expectunityWenlei=Markup(expectunityWenlei),
                           resultsTable=Markup(resultsTable.__html__()),
                           portfolioTable=Markup(portfolioTable.__html__()))


if __name__ == '__main__':
    app.run(debug=True)
