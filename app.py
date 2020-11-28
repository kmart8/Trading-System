from flask import Flask, render_template, Markup

# Kevin's files
from Kevin import *
from Kevin.backtest import *

# Wenlei's files
from Wenlei.TRIMA_wenlei_cao_get_backtrade_result import *

# Expectunity Calculation
from expectunityCalc import *


app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template("index.html")


@app.route("/results")
def results():
    backtest()
    expectunityKevin = combined_expectunity_calcuation(
        "/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/result.csv", 1000)
    main()
    expectunityWenlei = combined_expectunity_calcuation(
        "/Users/kevinmartin/Documents/Fall '20/GQP/Trading System/result.csv", 100)
    return render_template("results.html", expectunityKevin=Markup(expectunityKevin),
                           expectunityWenlei=Markup(expectunityWenlei))


if __name__ == '__main__':
    app.run(debug=True)
