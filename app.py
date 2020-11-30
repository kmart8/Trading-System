from flask import Flask, render_template, Markup
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

    backtest()
    expectunityKevin = combined_expectunity_calcuation(
        "/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/result.csv", 1000)
    main()
    expectunityWenlei = combined_expectunity_calcuation(
        "/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/result.csv", 100)
    # jackie()
    return render_template("results.html", expectunityKevin=Markup(expectunityKevin),
                           expectunityWenlei=Markup(expectunityWenlei))


if __name__ == '__main__':
    app.run(debug=True)
