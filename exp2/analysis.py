from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller, kpss


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
DATASETS = (
    ("E2_7.xlsx", "mortality", "Australian firearm-related homicide mortality rate", "Deaths per 100,000"),
    ("E2_8.xlsx", "wl", "Lake Michigan annual maximum water level", "Water level"),
)
AR_MODELS = (
    ("AR(1): xₜ = 0.8xₜ₋₁ + εₜ", (0.8,)),
    ("AR(1): xₜ = −0.7xₜ₋₁ + εₜ", (-0.7,)),
    ("AR(2): xₜ = −0.2xₜ₋₁ + 0.3xₜ₋₂ + εₜ", (-0.2, 0.3)),
    ("AR(2): xₜ = 0.2xₜ₋₁ − 0.3xₜ₋₂ + εₜ", (0.2, -0.3)),
)


def read_series(filename: str, value_column: str) -> pd.DataFrame:
    frame = pd.read_excel(ROOT / filename, sheet_name="Sheet1")
    frame["year"] = frame["year"].astype(int)
    return frame[["year", value_column]].dropna().sort_values("year")


def save_series_plots(frame: pd.DataFrame, column: str, title: str, ylabel: str, prefix: str) -> None:
    years = frame["year"]
    values = frame[column].astype(float)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(years, values, color="#245b8f", linewidth=1.35)
    ax.set(title=title, xlabel="Year", ylabel=ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{prefix}_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_acf(values, lags=24, alpha=0.05, ax=ax, zero=True)
    ax.set(title=f"Sample ACF: {title}", xlabel="Lag (years)", ylabel="Sample ACF")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{prefix}_acf.png", dpi=180)
    plt.close(fig)


def save_ar_plots() -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    for i, (title, coefficients) in enumerate(AR_MODELS):
        ar = [1.0, *(-coefficient for coefficient in coefficients)]
        process = ArmaProcess(ar=ar, ma=[1.0])
        rng = np.random.default_rng(20261009 + i)
        series = process.generate_sample(
            nsample=3000, burnin=500, distrvs=rng.standard_normal
        )
        plot_acf(series, lags=30, alpha=0.05, ax=axes[i, 0], zero=True)
        plot_pacf(series, lags=30, alpha=0.05, ax=axes[i, 1], method="ywm")
        axes[i, 0].set_title(f"{title} — ACF")
        axes[i, 1].set_title(f"{title} — PACF")
        for ax in axes[i]:
            ax.set_xlabel("Lag")
            ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Sample ACF and PACF of the four stationary AR models", y=1.005, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "ar_models_acf_pacf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def report_diagnostics(data: dict[str, pd.DataFrame]) -> None:
    for filename, column, title, _ in DATASETS:
        frame = data[filename]
        values = frame[column].astype(float)
        adf = adfuller(values, autolag="AIC")
        kpss_result = kpss(values, regression="c", nlags="auto")
        print(f"\n{title} ({frame['year'].min()}–{frame['year'].max()}, n={len(values)})")
        print(f"Mean={values.mean():.4f}; SD={values.std():.4f}; min={values.min():.4f}; max={values.max():.4f}")
        print(f"ADF: statistic={adf[0]:.4f}, p={adf[1]:.4f}")
        print(f"KPSS (constant): statistic={kpss_result[0]:.4f}, p={kpss_result[1]:.4f}")
        if filename == "E2_7.xlsx":
            lb = acorr_ljungbox(values, lags=[10], return_df=True).iloc[0]
            print(f"Ljung–Box (lag 10): Q={lb['lb_stat']:.4f}, p={lb['lb_pvalue']:.4g}")
        else:
            print("Ljung–Box is not used to classify this series as white noise because the level appears nonstationary.")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    data: dict[str, pd.DataFrame] = {}
    for filename, column, title, ylabel in DATASETS:
        frame = read_series(filename, column)
        data[filename] = frame
        save_series_plots(frame, column, title, ylabel, Path(filename).stem.lower())
    save_ar_plots()
    report_diagnostics(data)
    print(f"\nFigures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
