import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="ASX Strategy Lab", layout="wide")

st.title("🇦🇺 ASX Indicator Strategy Lab")
st.caption("Research, test and compare rule-based trading strategies.")

# ---------- Indicators ----------

def wma(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def hma(series, period=20):
    half = max(2, period // 2)
    root = max(2, int(np.sqrt(period)))
    return wma(2 * wma(series, half) - wma(series, period), root)

def kama(series, period=10, fast=2, slow=30):
    change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(period).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    result = pd.Series(np.nan, index=series.index)
    if len(series) <= period:
        return result
    result.iloc[period] = series.iloc[period]
    for i in range(period + 1, len(series)):
        result.iloc[i] = (
            result.iloc[i - 1]
            + sc.iloc[i] * (series.iloc[i] - result.iloc[i - 1])
        )
    return result

def efficiency_ratio(series, period=10):
    direction = series.diff(period).abs()
    noise = series.diff().abs().rolling(period).sum()
    return direction / noise.replace(0, np.nan)

def build_conditions(df):
    close = df["Close"]
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    h = hma(close, 20)
    k = kama(close, 10)
    er = efficiency_ratio(close, 10)

    return {
        "Close crosses above lower Bollinger Band":
            (close > lower) & (close.shift(1) <= lower.shift(1)),
        "Close crosses below upper Bollinger Band":
            (close < upper) & (close.shift(1) >= upper.shift(1)),
        "Close above Bollinger middle":
            close > middle,
        "Close below Bollinger middle":
            close < middle,
        "Close crosses above HMA":
            (close > h) & (close.shift(1) <= h.shift(1)),
        "Close crosses below HMA":
            (close < h) & (close.shift(1) >= h.shift(1)),
        "HMA rising":
            h > h.shift(3),
        "HMA falling":
            h < h.shift(3),
        "Close crosses above KAMA":
            (close > k) & (close.shift(1) <= k.shift(1)),
        "Close crosses below KAMA":
            (close < k) & (close.shift(1) >= k.shift(1)),
        "KAMA rising":
            k > k.shift(3),
        "KAMA falling":
            k < k.shift(3),
        "Efficiency Ratio above 0.4":
            er > 0.4,
        "Efficiency Ratio below 0.2":
            er < 0.2,
        "Positive 5-day momentum":
            close > close.shift(5),
        "Negative 5-day momentum":
            close < close.shift(5),
    }

# ---------- Backtesting ----------

def backtest(close, entry, exit_signal, fee, slippage):
    entry = entry.reindex(close.index).fillna(False).astype(bool)
    exit_signal = exit_signal.reindex(close.index).fillna(False).astype(bool)

    position = pd.Series(0.0, index=close.index)
    holding = 0
    for i in range(1, len(close)):
        if holding == 0 and entry.iloc[i - 1]:
            holding = 1
        elif holding == 1 and exit_signal.iloc[i - 1]:
            holding = 0
        position.iloc[i] = holding

    # Signals execute on the following bar to reduce look-ahead bias.
    daily_returns = close.pct_change().fillna(0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * (fee + slippage)
    strategy_returns = position * daily_returns - costs
    equity = (1 + strategy_returns).cumprod()

    total_return = equity.iloc[-1] - 1
    years = max(len(close) / 252, 1 / 252)
    cagr = equity.iloc[-1] ** (1 / years) - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe = (
        strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        if strategy_returns.std() > 0 else 0
    )
    drawdown = equity / equity.cummax() - 1
    trades = int((position.diff().fillna(position.iloc[0]) == 1).sum())

    return {
        "Return": total_return,
        "CAGR": cagr,
        "Max Drawdown": drawdown.min(),
        "Sharpe": sharpe,
        "Trades": trades,
        "Equity": equity,
        "Returns": strategy_returns,
    }

# ---------- User controls ----------

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Yahoo Finance ticker", "BHP.AX").upper().strip()
    style = st.selectbox(
        "Trading style",
        ["All three", "Long-term investing", "Swing trading", "Short-term trading"]
    )
    period = st.selectbox("Historical period", ["2y", "5y", "10y", "max"], index=1)
    interval = st.selectbox("Bar interval", ["1d", "1wk"], index=0)
    trials = st.slider("Strategies to test", 10, 200, 50, step=10)
    fee_pct = st.number_input("Fee per transaction (%)", 0.0, 2.0, 0.10, 0.01)
    slip_pct = st.number_input("Slippage per transaction (%)", 0.0, 2.0, 0.20, 0.01)
    min_trades = st.number_input("Minimum trades per test phase", 1, 100, 2)

st.info(
    "Examples: BHP.AX, CBA.AX, CSL.AX, RIO.AX, WDS.AX, TLS.AX, WES.AX. "
    "The index ticker ^AXJO may also be available."
)

if st.button("Download market data and search strategies", type="primary"):
    try:
        with st.spinner("Downloading historical data..."):
            data = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=True, progress=False
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()]
            data = data.dropna(subset=["Close"])
            close = data["Close"].astype(float)

        if len(data) < 150:
            st.error("Not enough historical data. Try a longer period or another ticker.")
            st.stop()

        conditions = build_conditions(data)
        condition_names = list(conditions.keys())

        if style == "Long-term investing":
            allowed = [n for n in condition_names if
                       "HMA" in n or "KAMA" in n or "momentum" in n]
        elif style == "Swing trading":
            allowed = condition_names
        elif style == "Short-term trading":
            allowed = [n for n in condition_names if
                       "Bollinger" in n or "Efficiency" in n or "momentum" in n]
        else:
            allowed = condition_names

        fee = fee_pct / 100
        slippage = slip_pct / 100
        # Chronological 60% / 20% / 20% split.
        cut1 = int(len(data) * 0.60)
        cut2 = int(len(data) * 0.80)
        phases = [
            ("In-sample", 0, cut1),
            ("Out-of-sample", cut1, cut2),
            ("Final holdout", cut2, len(data)),
        ]

        rows = []
        curves = {}
        random.seed(42)

        with st.spinner("Searching and validating strategies..."):
            for number in range(trials):
                entry_name, exit_name = random.sample(allowed, 2)
                entry = conditions[entry_name]
                exit_signal = conditions[exit_name]
                phase_results = []
                valid = True

                for phase_name, start, end in phases:
                    # Extra history is retained for indicator warm-up.
                    start_eval = max(0, start)
                    c = close.iloc[start_eval:end]
                    e = entry.iloc[start_eval:end]
                    x = exit_signal.iloc[start_eval:end]
                    result = backtest(c, e, x, fee, slippage)

                    # Exclude the warm-up portion from phase scoring.
                    if start > 0:
                        warm = min(30, len(c) - 1)
                        c2 = c.iloc[warm:]
                        e2 = e.iloc[warm:]
                        x2 = x.iloc[warm:]
                        result = backtest(c2, e2, x2, fee, slippage)

                    phase_results.append(result)
                    if result["Return"] <= 0 or result["Trades"] < min_trades:
                        valid = False

                rows.append({
                    "Strategy": f"S{number + 1}",
                    "Entry condition": entry_name,
                    "Exit condition": exit_name,
                    "In-sample return %": phase_results[0]["Return"] * 100,
                    "Out-of-sample return %": phase_results[1]["Return"] * 100,
                    "Holdout return %": phase_results[2]["Return"] * 100,
                    "Holdout max drawdown %": phase_results[2]["Max Drawdown"] * 100,
                    "Holdout Sharpe": phase_results[2]["Sharpe"],
                    "Holdout trades": phase_results[2]["Trades"],
                    "Passed all phases": valid,
                })
                curves[f"S{number + 1}"] = phase_results[2]["Equity"]

        results = pd.DataFrame(rows).sort_values(
            ["Passed all phases", "Holdout return %"], ascending=[False, False]
        ).reset_index(drop=True)

        st.session_state["results"] = results
        st.session_state["curves"] = curves
        st.session_state["ticker"] = ticker
        st.session_state["data"] = data
        st.session_state["conditions"] = conditions

    except Exception as exc:
        st.error(f"Could not complete the search: {exc}")
        st.caption("Check the ticker, data availability and internet connection.")

if "results" in st.session_state:
    results = st.session_state["results"]
    st.subheader(f"Results for {st.session_state['ticker']}")
    passed = results[results["Passed all phases"]]
    st.metric("Strategies passing all three phases", len(passed))
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download results as CSV",
        results.to_csv(index=False),
        file_name="asx_strategy_results.csv",
        mime="text/csv",
    )

    if len(passed):
        selected = st.selectbox("View holdout equity curve", passed["Strategy"].tolist())
        curve = st.session_state["curves"][selected]
        fig, ax = plt.subplots()
        ax.plot(curve.index, curve.values)
        ax.set_title(f"{selected} — final holdout equity")
        ax.set_ylabel("Growth of $1")
        ax.grid(True)
        st.pyplot(fig)
        row = passed[passed["Strategy"] == selected].iloc[0]
        st.write("Entry:", row["Entry condition"])
        st.write("Exit:", row["Exit condition"])
    else:
        st.warning(
            "No strategy passed every filter. This is a normal research outcome; "
            "do not loosen the filters only to force a positive result."
        )

st.caption(
    "Educational research prototype only. Historical results do not predict future "
    "returns. This starter uses a representative condition set, not the complete "
    "93-condition library from the reference document."
)
