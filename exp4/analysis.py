"""实验四：平稳性、ARMA 模型比较、残差诊断及未来五年预测。"""
from itertools import combinations
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


ROOT = Path(__file__).resolve().parent
FIGURE = ROOT / "figure"
RESULTS = ROOT / "results"
ALPHA = 0.05
LAGS = 20
RESIDUAL_LAGS = [6, 10]
AICC_TOLERANCE = 2.0
ORDERS = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2),
          (0, 3), (1, 1), (1, 2), (2, 1)]
DATASETS = [
    ("e4_1", "公司月度盈亏", "月", 50, 60),
    ("e4_2", "城市月度人口净流入量", "月", 48, 60),
    ("e4_4", "城市年度人口死亡率", "年", 45, 5),
]
BLUE, ORANGE = "#245b8f", "#b96128"


def save_csv(frame, name):
    frame.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.12g")


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE / f"{name}.png", dpi=180)
    plt.close(fig)


def diagnostics(values, regression="c"):
    adf = adfuller(values, regression=regression, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kp = kpss(values, regression=regression, nlags="auto")
    lb = acorr_ljungbox(values, lags=[1, 6, 10], return_df=True)
    return {
        "n": len(values), "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "regression": regression,
        "adf_stat": float(adf[0]), "adf_p": float(adf[1]),
        "adf_lags": int(adf[2]), "adf_nobs": int(adf[3]),
        "kpss_stat": float(kp[0]), "kpss_p": float(kp[1]),
        "kpss_p_bound": ">" if kp[1] == 0.1 else "<" if kp[1] == 0.01 else "=",
        "kpss_lags": int(kp[2]),
        **{f"lb{lag}_{key}": float(lb.loc[lag, col])
           for lag in [1, 6, 10]
           for key, col in [("stat", "lb_stat"), ("p", "lb_pvalue")]},
    }


def correlations(values):
    a, ai = acf(values, nlags=LAGS, alpha=ALPHA, fft=False,
                adjusted=False, bartlett_confint=True)
    p, pi = pacf(values, nlags=LAGS, alpha=ALPHA, method="ywm")
    return pd.DataFrame({
        "lag": np.arange(1, LAGS + 1), "acf": a[1:], "pacf": p[1:],
        "acf_ci_lower": ai[1:, 0], "acf_ci_upper": ai[1:, 1],
        "pacf_ci_lower": pi[1:, 0], "pacf_ci_upper": pi[1:, 1],
        "acf_significant": (ai[1:, 0] > 0) | (ai[1:, 1] < 0),
        "pacf_significant": (pi[1:, 0] > 0) | (pi[1:, 1] < 0),
    })


def plot_correlations(corr, title, unit, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in zip(axes, ["acf", "pacf"], ["样本 ACF", "样本 PACF"]):
        lags, values = corr.lag.to_numpy(), corr[key].to_numpy()
        ax.fill_between(lags, corr[f"{key}_ci_lower"] - values,
                        corr[f"{key}_ci_upper"] - values, color="#dce8f4")
        ax.vlines(lags, 0, values, color=BLUE)
        ax.plot(lags, values, "o", color=BLUE, ms=3.5)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set(title=label, xlabel=f"滞后阶数（{unit}）", ylabel="相关系数",
               xlim=(0.4, LAGS + 0.6), ylim=(-1.05, 1.05))
        ax.set_xticks([1, 4, 8, 12, 16, 20])
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title + "（阴影为95%逐阶参考区间）", fontsize=12)
    save_figure(fig, name)


def model_label(p, q, trend):
    if isinstance(q, list):
        base = "MA{" + ",".join(map(str, q)) + "}"
    elif p == q == 0:
        base = "白噪声"
    elif q == 0:
        base = f"AR({p})"
    elif p == 0:
        base = f"MA({q})"
    else:
        base = f"ARMA({p},{q})"
    return base + {"n": "（零均值）", "c": " + 均值", "ct": " + 线性趋势"}[trend]


def fit_candidates(values, key):
    specs = [(p, q, tr) for tr in (["n", "c"] if key == "e4_1" else
                                  ["c", "ct"] if key == "e4_2" else ["c"])
             for p, q in ORDERS]
    if key == "e4_4":
        # 系统比较 1—3 阶 MA 的所有非空滞后子集，考察固定零系数的优化。
        for size in range(1, 4):
            for subset in combinations([1, 2, 3], size):
                if subset != tuple(range(1, max(subset) + 1)):
                    specs.append((0, list(subset), "c"))
    rows, fits = [], {}
    for p, q, trend in specs:
        label = model_label(p, q, trend)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = ARIMA(values, order=(p, 0, q), trend=trend,
                        enforce_stationarity=True, enforce_invertibility=True).fit(
                            method_kwargs={"maxiter": 500})
        fits[label] = fit
        # 稀疏模型按实际估计的 AR/MA 系数数量扣除自由度。
        dynamic_df = len(fit.arparams) + len(fit.maparams)
        innovation = fit.filter_results.standardized_forecasts_error[0]
        lb = acorr_ljungbox(innovation, lags=RESIDUAL_LAGS, model_df=dynamic_df)
        converged = bool(fit.mle_retvals["converged"])
        row = {
            "model": label, "ar_lags": str(list(range(1, p + 1))),
            "ma_lags": str(q if isinstance(q, list) else list(range(1, q + 1))),
            "trend": trend, "n_parameters": len(fit.params),
            "dynamic_df": dynamic_df, "log_likelihood": float(fit.llf),
            "aic": float(fit.aic), "aicc": float(fit.aicc), "bic": float(fit.bic),
            "converged": converged,
            "min_ar_root_modulus": float(min(abs(fit.arroots))) if len(fit.arroots) else None,
            "min_ma_root_modulus": float(min(abs(fit.maroots))) if len(fit.maroots) else None,
            "residual_lb6_p": float(lb.loc[6, "lb_pvalue"]),
            "residual_lb10_p": float(lb.loc[10, "lb_pvalue"]),
            "diagnostics_pass": bool(converged and np.isfinite(fit.aicc)
                                     and (lb.lb_pvalue > ALPHA).all()),
            "warnings": " | ".join(sorted({str(w.message) for w in caught})),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table.loc[table.diagnostics_pass]
    if eligible.empty:
        raise ValueError(f"{key}: 没有模型通过预设的残差检验，请检查候选范围")
    best_aicc = eligible.aicc.min()
    table["delta_aicc_eligible"] = table.aicc - best_aicc
    table["within_2_of_eligible_best"] = (
        table.diagnostics_pass & (table.delta_aicc_eligible <= AICC_TOLERANCE))
    shortlist = table.loc[table.within_2_of_eligible_best]
    chosen = shortlist.sort_values(["n_parameters", "aicc"]).iloc[0]["model"]
    table["selected"] = table.model == chosen
    return table.sort_values("aicc").reset_index(drop=True), fits, chosen


def forecast_frame(fit, n, steps):
    forecast = fit.get_forecast(steps=steps)
    interval = np.asarray(forecast.conf_int(alpha=ALPHA))
    return pd.DataFrame({"horizon": np.arange(1, steps + 1),
                         "t": np.arange(n + 1, n + steps + 1),
                         "forecast": np.asarray(forecast.predicted_mean),
                         "lower_95": interval[:, 0], "upper_95": interval[:, 1]})


def plot_series(frame, title, unit, key, fitted, forecast, alternative=None):
    t, x = frame.t, frame.x
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x, "o-", color=BLUE, ms=3, lw=1.3, label="观测值")
    ax.axhline(x.mean(), color="0.5", ls="--", lw=1, label="样本均值")
    if key == "e4_2":
        slope, intercept = np.polyfit(t, x, 1)
        ax.plot(t, intercept + slope * t, color=ORANGE, label="线性趋势参考")
    ax.set(title=title, xlabel=f"时间序号 t（{unit}）", ylabel="原数据单位")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    save_figure(fig, f"{key}_time")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x, color=BLUE, lw=1.4, label="观测值")
    ax.plot(t, fitted, color=ORANGE, lw=1.3, ls="--", label="样本内一步拟合")
    ax.set(title=title + "拟合", xlabel=f"时间序号 t（{unit}）", ylabel="原数据单位")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    save_figure(fig, f"{key}_fit")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, x, color=BLUE, lw=1.3, label="历史观测")
    ax.plot(np.r_[t.iloc[-1], forecast.t], np.r_[x.iloc[-1], forecast.forecast],
            color=ORANGE, lw=1.6, label="未来5年点预测")
    ax.fill_between(forecast.t, forecast.lower_95, forecast.upper_95,
                    color="#f0d9c5", alpha=0.8, label="条件95%逐点预测区间")
    if alternative is not None:
        ax.plot(alternative.t, alternative.forecast, color="#46745b", ls="--",
                label="常均值 MA(1) 对照")
    ax.axvline(t.iloc[-1] + 0.5, color="0.4", ls=":")
    ax.set(title=title + "未来5年预测", xlabel=f"时间序号 t（{unit}）", ylabel="原数据单位")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, f"{key}_forecast")


def plot_residuals(innovation, title, unit, key):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(np.arange(1, len(innovation) + 1), innovation, color=BLUE)
    axes[0, 0].axhline(0, color="0.4", lw=0.8)
    axes[0, 0].set(title="标准化一步预测误差", xlabel=f"时间序号 t（{unit}）")
    axes[0, 1].hist(innovation, bins=10, density=True, color="#aac4de", edgecolor="white")
    grid = np.linspace(-3.5, 3.5, 200)
    axes[0, 1].plot(grid, stats.norm.pdf(grid), color=ORANGE, label="标准正态密度")
    axes[0, 1].set(title="误差分布", xlabel="标准化误差", ylabel="密度")
    axes[0, 1].legend(frameon=False, fontsize=9)
    (theory, ordered), (slope, intercept, _) = stats.probplot(innovation, dist="norm")
    axes[1, 0].plot(theory, ordered, "o", color=BLUE, ms=4)
    axes[1, 0].plot(theory, intercept + slope * theory, color=ORANGE)
    axes[1, 0].set(title="正态 Q–Q 图", xlabel="正态理论分位数", ylabel="样本分位数")
    lags = np.arange(1, LAGS + 1)
    coeff = acf(innovation, nlags=LAGS, fft=False)[1:]
    bound = stats.norm.ppf(1 - ALPHA / 2) / np.sqrt(len(innovation))
    ax = axes[1, 1]
    ax.axhspan(-bound, bound, color="#dce8f4")
    ax.vlines(lags, 0, coeff, color=BLUE)
    ax.plot(lags, coeff, "o", color=BLUE, ms=3)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(title="标准化误差 ACF（95%白噪声参考带）", xlabel=f"滞后阶数（{unit}）",
           ylabel="相关系数", ylim=(-1.05, 1.05))
    ax.set_xticks([1, 4, 8, 12, 16, 20])
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    fig.suptitle(title + "残差诊断", fontsize=13)
    save_figure(fig, f"{key}_residuals")


def plot_model_comparison(table):
    fig, ax = plt.subplots(figsize=(11, 7))
    shown = table.sort_values("aicc", ascending=False)
    colors = [ORANGE if r.selected else BLUE if r.diagnostics_pass else "#b3b3b3"
              for r in shown.itertuples()]
    ax.barh(shown.model, shown.aicc - table.aicc.min(), color=colors)
    selected = shown.loc[shown.selected].iloc[0]
    ax.scatter([selected.aicc - table.aicc.min()], [selected.model],
               color=ORANGE, s=65, zorder=3, clip_on=False)
    ax.set(title="人口死亡率候选模型比较", xlabel="AICc 与全部候选最小值之差（越小越好）")
    ax.grid(axis="x", alpha=0.2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="所选模型"),
                       Patch(color=BLUE, label="残差检验通过"),
                       Patch(color="#b3b3b3", label="残差检验未通过")], frameon=False)
    save_figure(fig, "e4_4_model_comparison")


def main():
    FIGURE.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    output, diagnostic_rows = {}, []
    for key, title, unit, n, steps in DATASETS:
        frame = pd.read_csv(ROOT / "doc" / f"{key.upper()}.csv")
        if (frame.shape != (n, 2) or frame.isna().any().any()
                or not np.array_equal(frame.t, np.arange(1, n + 1))):
            raise ValueError(f"{key}: 数据行数、连续序号或缺失值异常")
        values = frame.x.to_numpy(dtype=float)
        diag = diagnostics(values)
        diagnostic_rows.append({"series": key, **diag})
        corr = correlations(values)
        save_csv(corr, f"{key}_correlations")
        plot_correlations(corr, title, unit, f"{key}_acf_pacf")

        table, fits, chosen = fit_candidates(values, key)
        save_csv(table, f"{key}_model_comparison")
        fit = fits[chosen]
        selected = {k: None if pd.isna(v) else v
                    for k, v in table.loc[table.selected].iloc[0].to_dict().items()}
        parameters = pd.DataFrame({"parameter": fit.param_names, "estimate": fit.params,
                                   "std_error": fit.bse, "p_value": fit.pvalues,
                                   "lower_95": fit.conf_int()[:, 0],
                                   "upper_95": fit.conf_int()[:, 1]})
        save_csv(parameters, f"{key}_parameters")
        fitted = np.asarray(fit.fittedvalues)
        innovation = fit.filter_results.standardized_forecasts_error[0]
        save_csv(frame.assign(fitted=fitted, residual=fit.resid,
                              standardized_innovation=innovation), f"{key}_fitted")
        lb = acorr_ljungbox(innovation, lags=RESIDUAL_LAGS,
                           model_df=int(selected["dynamic_df"]))
        save_csv(lb.rename_axis("lag").reset_index(), f"{key}_residual_ljung_box")
        forecast = forecast_frame(fit, n, steps)
        save_csv(forecast, f"{key}_forecast")
        alternative = None
        if key == "e4_2":
            alternative = forecast_frame(fits[model_label(0, 1, "c")], n, steps)
            save_csv(alternative, "e4_2_constant_forecast")
            # 使用含趋势项的原序列检验，避免把估计趋势当成事先已知。
            trend_diag = diagnostics(values, regression="ct")
            diagnostic_rows.append({"series": "e4_2_trend_stationarity", **trend_diag})
            params = dict(zip(fit.param_names, fit.params))
            detrended = values - params["const"] - params["x1"] * frame.t.to_numpy()
            det_corr = correlations(detrended)
            save_csv(det_corr, "e4_2_detrended_correlations")
            plot_correlations(det_corr, title + "去线性趋势后", unit, "e4_2_detrended_acf_pacf")
        if key == "e4_4":
            plot_model_comparison(table)
        plot_series(frame, title, unit, key, fitted, forecast, alternative)
        plot_residuals(innovation, title, unit, key)
        jb = stats.jarque_bera(innovation)
        output[key] = {
            "title": title, "time_unit": unit, "forecast_steps": steps,
            "diagnostics": diag, "selected_model": selected,
            "parameters": parameters.to_dict(orient="records"),
            "in_sample_one_step_rmse_after_t3": float(np.sqrt(np.mean((values[3:] - fitted[3:]) ** 2))),
            "innovation_jb_stat": float(jb.statistic), "innovation_jb_p": float(jb.pvalue),
            "forecast": forecast.to_dict(orient="records"),
        }
        print(f"{key}: {chosen}; AICc={fit.aicc:.4f}; "
              f"residual LB(10) p={selected['residual_lb10_p']:.4f}; forecast {steps} {unit}", flush=True)
    save_csv(pd.DataFrame(diagnostic_rows), "diagnostics")
    output["settings"] = {
        "alpha": ALPHA, "correlation_lags": LAGS, "acf_ci": "Bartlett",
        "pacf_method": "ywm", "adf_autolag": "AIC", "kpss_nlags": "auto",
        "residual_test": "standardized one-step innovations; model_df = estimated AR + MA count",
        "residual_lags": RESIDUAL_LAGS, "aicc_tolerance": AICC_TOLERANCE,
        "selection": "converged and residual LB6/LB10 p > .05; within 2 AICc, fewest parameters, then AICc",
        "interval": "pointwise Gaussian prediction interval conditional on estimated parameters",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__},
    }
    (RESULTS / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("分析完成：exp4/figure/、exp4/results/。运行方式与解释见 exp4/README.md。")


if __name__ == "__main__":
    main()
