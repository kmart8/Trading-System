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
# from Jackie.Turner_TradingSystemWithBacktrader import *

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
    r.append(Result("AMD", 30000, 25000, 5000, 0.05, expectunityKevin))

    main()
    expectunityWenlei = combined_expectunity_calcuation(
        "/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/result.csv", 100)
    r.append(Result("Mutual Funds", 30000, 46758,
                    16758, 16758/30000, expectunityWenlei))
    # jackie()

    resultsTable = ResultsTable(r)
    return render_template("results.html", expectunityKevin=Markup(expectunityKevin),
                           expectunityWenlei=Markup(expectunityWenlei),
                           resultsTable=Markup(resultsTable.__html__()))


if __name__ == '__main__':
    app.run(debug=True)
