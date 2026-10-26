# asx-indicator-strategy-labs
ASX Indicator Strategy Research Platform
# 🇦🇺 ASX Indicator Strategy Lab

A Python-based research application for testing technical indicator trading strategies on Australian Securities Exchange (ASX) stocks using historical market data.

## Features

* Download historical market data using Yahoo Finance.
* Test ASX stocks such as BHP.AX, CBA.AX, CSL.AX and RIO.AX.
* Explore long-term, swing-trading and short-term strategy styles.
* Use Bollinger Bands, Hull Moving Average (HMA), Kaufman Adaptive Moving Average (KAMA) and Efficiency Ratio indicators.
* Generate and backtest candidate strategies.
* Compare in-sample, out-of-sample and final holdout performance.
* Review returns, drawdowns, Sharpe ratios and trade counts.
* Export backtest results as CSV.

## Requirements

* Python 3.10 or a compatible version.
* Streamlit
* yfinance
* pandas
* numpy
* matplotlib

## Run locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Deploy online

Deploy this repository using Streamlit Community Cloud.

Select:

* Repository: `asx-indicator-strategy-labs`
* Branch: `main`
* Main file path: `app.py`

## Disclaimer

This application is intended for educational and research purposes only. Historical backtest results do not guarantee future performance. Transaction costs, slippage, market liquidity, corporate actions and data quality can affect results. Verify all strategies independently before making investment decisions.
