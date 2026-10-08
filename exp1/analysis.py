from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "doc"
FIGURES = ROOT / "figures"


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


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    co2 = load_series("E2_2.xlsx", "co2")
    sales = load_series("E2_5.xlsx", "x")

    save_time_plot(co2, "Monthly atmospheric CO₂, 1975–1980", "CO₂ concentration", "co2_time.png")
    save_acf_plot(co2, "Sample ACF of monthly atmospheric CO₂", "co2_acf.png", lags=24)
    save_time_plot(sales, "Monthly company sales, 2000–2003", "Sales", "sales_time.png")
    save_acf_plot(sales, "Sample ACF of monthly company sales", "sales_acf.png", lags=24)

    result = acorr_ljungbox(sales, lags=[10], return_df=True)
    print("Sales series Ljung–Box test (lag 10)")
    print(result.to_string())
    print(f"Figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
