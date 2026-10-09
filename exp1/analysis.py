import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "doc"
FIGURES = ROOT / "figures"
APPLE_FILE = DATA / "AAPL_2020_2025.csv"
APPLE_SOURCE = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"


def load_apple(refresh: bool = False) -> pd.DataFrame:
    """Download once, then use the saved daily Yahoo Finance data snapshot."""
    if refresh or not APPLE_FILE.exists():
        params = {
            "period1": int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()),
            "period2": int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()),
            "interval": "1d",
            "events": "div,splits",
        }
        response = requests.get(
            APPLE_SOURCE, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
        )
        response.raise_for_status()
        chart = response.json()["chart"]
        if chart["error"]:
            raise RuntimeError(chart["error"])
        data = chart["result"][0]
        dates = (
            pd.to_datetime(data["timestamp"], unit="s", utc=True)
            .tz_convert(data["meta"]["exchangeTimezoneName"])
            .tz_localize(None)
            .normalize()
        )
        quote = data["indicators"]["quote"][0]
        frame = pd.DataFrame(
            {
                "Open": quote["open"],
                "High": quote["high"],
                "Low": quote["low"],
                "Close": quote["close"],
                "Adj Close": data["indicators"]["adjclose"][0]["adjclose"],
                "Volume": quote["volume"],
            }, index=dates,
        ).sort_index()
        frame.index.name = "Date"
        frame.to_csv(APPLE_FILE, float_format="%.8f")
        metadata = {
            "symbol": "AAPL",
            "source": "Yahoo Finance",
            "source_page": "https://finance.yahoo.com/quote/AAPL/history/",
            "download_url": response.url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_start": "2020-01-01",
            "requested_end_exclusive": "2026-01-01",
            "currency": data["meta"]["currency"],
            "exchange_timezone": data["meta"]["exchangeTimezoneName"],
            "price_column": "Adj Close",
            "adjustment": "Adjusted for stock splits and dividends by Yahoo Finance",
            "sha256": hashlib.sha256(APPLE_FILE.read_bytes()).hexdigest(),
        }
        (DATA / "AAPL_2020_2025_source.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return pd.read_csv(APPLE_FILE, index_col="Date", parse_dates=["Date"])


def load_series(filename: str, value_column: str) -> pd.Series:
    frame = pd.read_excel(DATA / filename, sheet_name="Sheet1", parse_dates=["time"])
    return frame.set_index("time")[value_column].astype(float)


def save_time_plot(series: pd.Series, title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(series.index, series.values, color="#245b8f", linewidth=1.5)
    ax.set(title=title, xlabel="Time", ylabel=ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160)
    plt.close(fig)


def save_acf_plot(series: pd.Series, title: str, filename: str, lags: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_acf(series, lags=lags, alpha=0.05, ax=ax, zero=True)
    ax.set(title=title, xlabel="Lag", ylabel="Sample ACF")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160)
    plt.close(fig)


def describe_series(series: pd.Series) -> dict:
    """Record the values used in the report without rounding the source results."""
    correlations = acf(series, nlags=24, fft=False)
    return {
        "count": int(series.size),
        "start": series.index.min().strftime("%Y-%m"),
        "end": series.index.max().strftime("%Y-%m"),
        "missing": int(series.isna().sum()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "annual_mean": {
            str(year): float(values.mean())
            for year, values in series.groupby(series.index.year)
        },
        "acf": [float(value) for value in correlations],
    }


def stationarity_test(series: pd.Series) -> dict:
    statistic, p_value, used_lag, nobs, critical, _ = adfuller(
        series, regression="c", autolag="AIC"
    )
    return {
        "statistic": float(statistic), "p_value": float(p_value),
        "used_lag": int(used_lag), "nobs": int(nobs),
        "critical_values": {key: float(value) for key, value in critical.items()},
        "regression": "constant", "autolag": "AIC",
    }


def analyze_apple(frame: pd.DataFrame) -> dict:
    price = frame["Adj Close"]
    if frame.index.has_duplicates or price.isna().any() or (price <= 0).any():
        raise ValueError("Apple prices must have unique dates and positive, complete values")
    returns = np.log(price).diff().dropna() * 100
    monthly = price.resample("ME").last().rename("Adj Close")
    monthly.to_csv(DATA / "AAPL_monthly_close.csv", index_label="Date", float_format="%.8f")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(price, color="#245b8f", linewidth=1, label="Adjusted close")
    ax.plot(price.rolling(20).mean(), linewidth=1, label="20 trading-day mean")
    ax.plot(price.rolling(60).mean(), linewidth=1, label="60 trading-day mean")
    ax.set(title="Apple daily adjusted close, 2020-2025", xlabel="Date", ylabel="USD")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "apple_price_time.png", dpi=160)
    plt.close(fig)
    save_acf_plot(price, "Sample ACF of Apple adjusted close", "apple_price_acf.png", lags=40)
    save_time_plot(returns, "Apple daily log returns, 2020-2025", "Log return (%)", "apple_returns_time.png")
    save_acf_plot(returns, "Sample ACF of Apple daily log returns", "apple_returns_acf.png", lags=40)

    lb = acorr_ljungbox(returns, lags=[10], return_df=True)
    squared_lb = acorr_ljungbox(returns ** 2, lags=[10], return_df=True)
    summary = describe_series(price)
    summary.update({
        "first_date": price.index[0].strftime("%Y-%m-%d"),
        "last_date": price.index[-1].strftime("%Y-%m-%d"),
        "first_price": float(price.iloc[0]), "last_price": float(price.iloc[-1]),
        "min_date": price.idxmin().strftime("%Y-%m-%d"),
        "max_date": price.idxmax().strftime("%Y-%m-%d"),
        "return_count": int(returns.size), "return_mean_pct": float(returns.mean()),
        "return_std_pct": float(returns.std()), "return_acf": acf(returns, nlags=40).tolist(),
        "monthly_count": int(monthly.size), "missing_by_column": frame.isna().sum().to_dict(),
        "duplicate_dates": int(frame.index.duplicated().sum()),
        "price_adf": stationarity_test(price), "return_adf": stationarity_test(returns),
        "return_ljung_box": {"lag": 10, "statistic": float(lb.loc[10, "lb_stat"]), "p_value": float(lb.loc[10, "lb_pvalue"])},
        "squared_return_ljung_box": {"lag": 10, "statistic": float(squared_lb.loc[10, "lb_stat"]), "p_value": float(squared_lb.loc[10, "lb_pvalue"])},
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 time series analysis")
    parser.add_argument("--refresh-apple", action="store_true", help="Refresh the saved Yahoo Finance snapshot")
    args = parser.parse_args()
    FIGURES.mkdir(exist_ok=True)

    co2 = load_series("E2_2.xlsx", "co2")
    sales = load_series("E2_5.xlsx", "x")

    save_time_plot(co2, "Monthly atmospheric CO₂, 1975–1980", "CO₂ concentration", "co2_time.png")
    save_acf_plot(co2, "Sample ACF of monthly atmospheric CO₂", "co2_acf.png", lags=24)
    save_time_plot(sales, "Monthly company sales, 2000–2003", "Sales", "sales_time.png")
    save_acf_plot(sales, "Sample ACF of monthly company sales", "sales_acf.png", lags=24)

    result = acorr_ljungbox(sales, lags=[10], return_df=True)
    results = {
        "apple": analyze_apple(load_apple(args.refresh_apple)),
        "co2": describe_series(co2),
        "sales": describe_series(sales),
        "sales_ljung_box": {
            "lag": 10,
            "statistic": float(result.loc[10, "lb_stat"]),
            "p_value": float(result.loc[10, "lb_pvalue"]),
            "alpha": 0.05,
            "model_df": 0,
        },
    }
    (ROOT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Sales series Ljung–Box test (lag 10)")
    print(result.to_string())
    print("Apple ADF p-values:", results["apple"]["price_adf"]["p_value"], results["apple"]["return_adf"]["p_value"])
    print(f"Figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
