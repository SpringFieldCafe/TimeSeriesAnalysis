"""实验五：十条序列的平稳性、ARIMA 建模及题目指定期限预测。"""
from pathlib import Path
import json
import platform
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
FIGURES, RESULTS = ROOT / "figure", ROOT / "results"
ALPHA, ADF_MAXLAG, AICC_TOLERANCE = 0.05, 4, 2.0
RESIDUAL_LAGS = [6, 10]
ORDERS = [(p, q) for p in range(5) for q in range(5) if p + q <= 4]
BLUE, ORANGE = "#245b8f", "#b96128"
SPECS = [
    ("e4_8", "E4_8", "x", "连续生产记录", 201, 1, "记录序号"),
    ("sheep", "E5_3", "number", "英国绵羊数量", 72, 7, "年"),
    ("birth_rate", "E5_4", "birth_rate", "人口出生率", 38, 10, "年"),
    ("mortality", "E5_4", "mortality", "人口死亡率", 38, 10, "年"),
    ("ngr", "E5_4", "ngr", "人口自然增长率", 38, 10, "年"),
    ("maize_price", "E5_5", "maize_price", "玉米销售价格", 81, 10, "年"),
    ("maize_yield", "E5_5", "maize_yield", "玉米产量", 81, 10, "年"),
    ("wages", "E5_5", "wages", "农场工人平均工资", 81, 10, "年"),
    ("pig_price", "E5_5", "pig_price", "生猪销售价格", 81, 10, "年"),
    ("pig_yield", "E5_5", "pig_yield", "生猪产量", 81, 10, "年"),
]


def save_csv(frame, name):
    frame.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.12g")


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


def diagnostic(values, regression="c"):
    primary = adfuller(values, maxlag=ADF_MAXLAG, regression=regression, autolag="AIC")
    default = adfuller(values, regression=regression, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kp = kpss(values, regression=regression, nlags="auto")
    lb = acorr_ljungbox(values, lags=[1, 6, 10], return_df=True)
    return {
        "n": len(values), "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)), "regression": regression,
        "adf_stat": float(primary[0]), "adf_p": float(primary[1]),
        "adf_lags": int(primary[2]), "adf_nobs": int(primary[3]),
        "adf_default_stat": float(default[0]), "adf_default_p": float(default[1]),
        "adf_default_lags": int(default[2]),
        "kpss_stat": float(kp[0]), "kpss_p": float(kp[1]),
        "kpss_p_bound": ">" if kp[1] == 0.1 else "<" if kp[1] == 0.01 else "=",
        "kpss_lags": int(kp[2]),
        **{f"lb{lag}_{key}": float(lb.loc[lag, column])
           for lag in [1, 6, 10]
           for key, column in [("stat", "lb_stat"), ("p", "lb_pvalue")]},
    }


def correlations(values):
    nlags = min(20, len(values) // 2 - 1)
    a, ai = acf(values, nlags=nlags, alpha=ALPHA, fft=False, adjusted=False)
    p, pi = pacf(values, nlags=nlags, alpha=ALPHA, method="ywm")
    return pd.DataFrame({
        "lag": np.arange(1, nlags + 1), "acf": a[1:], "pacf": p[1:],
        "acf_ci_lower": ai[1:, 0], "acf_ci_upper": ai[1:, 1],
        "pacf_ci_lower": pi[1:, 0], "pacf_ci_upper": pi[1:, 1],
    })


def plot_correlations(corr, title, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key in zip(axes, ["acf", "pacf"]):
        lag, value = corr.lag.to_numpy(), corr[key].to_numpy()
        ax.fill_between(lag, corr[f"{key}_ci_lower"] - value,
                        corr[f"{key}_ci_upper"] - value, color="#dce8f4")
        ax.vlines(lag, 0, value, color=BLUE)
        ax.plot(lag, value, "o", ms=3.5, color=BLUE)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set(title=key.upper(), xlabel="滞后阶数", ylabel="相关系数", ylim=(-1.05, 1.05))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title + "（阴影为95%逐阶参考区间）", fontsize=12)
    save_figure(fig, name)


def model_label(p, d, q, trend):
    return f"ARIMA({p},{d},{q})" + {
        "n": "（无均值项）" if d == 0 else "（无漂移）",
        "c": "（含均值）", "t": "（含漂移）",
    }[trend]


def fit_candidates(values, d):
    # 统一缩放至样本标准差为1，改善不同数量级序列的数值优化；输出还原到原数据尺度。
    center, scale = float(np.mean(values)), float(np.std(values, ddof=1))
    scaled = (values - center) / scale
    trends = ["n", "c"] if d == 0 else ["n", "t"]
    # d=0 的无均值模型必须使用未中心化数据，避免把样本均值作为已知常数引入。
    rows, fits = [], {}
    for p, q in ORDERS:
        for trend in trends:
            label = model_label(p, d, q, trend)
            offset = 0.0 if d == 0 and trend == "n" else center
            z = scaled if offset == center else values / scale
            row = {"model": label, "p": p, "d": d, "q": q, "trend": trend}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    fit = ARIMA(z, order=(p, d, q), trend=trend,
                                enforce_stationarity=True, enforce_invertibility=True).fit(
                                    method_kwargs={"maxiter": 500})
                    burn = int(fit.loglikelihood_burn)
                    innovation = fit.filter_results.standardized_forecasts_error[0, burn:]
                    lb = acorr_ljungbox(innovation, lags=RESIDUAL_LAGS, model_df=p + q)
                    # 对缩放后的似然补回尺度常数，使数值对应原数据单位。
                    n_effective = len(values) - burn
                    adjustment = 2 * n_effective * np.log(scale)
                    aicc = float(fit.aicc + adjustment)
                    roots = np.r_[np.abs(fit.arroots), np.abs(fit.maroots)]
                    root_min = float(np.min(roots)) if len(roots) else None
                    converged = bool(fit.mle_retvals["converged"])
                    row.update({
                        "k": len(fit.params), "n_effective": n_effective,
                        "loglikelihood": float(fit.llf - adjustment / 2),
                        "aic": float(fit.aic + adjustment), "aicc": aicc,
                        "bic": float(fit.bic + adjustment), "converged": converged,
                        "min_root_modulus": root_min,
                        "lb6_p": float(lb.loc[6, "lb_pvalue"]),
                        "lb10_p": float(lb.loc[10, "lb_pvalue"]),
                        "eligible": bool(converged and np.isfinite(aicc)
                                         and (lb.lb_pvalue > ALPHA).all()),
                    })
                    fits[label] = (fit, offset, scale)
                except (ValueError, np.linalg.LinAlgError) as error:
                    row.update({"eligible": False, "converged": False, "error": str(error)})
            row["warnings"] = " | ".join(sorted({str(w.message) for w in caught}))
            rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table[table.eligible]
    if eligible.empty:
        raise RuntimeError("没有候选模型通过收敛和残差诊断，请检查模型比较表。")
    best = eligible.aicc.min()
    chosen = eligible[eligible.aicc <= best + AICC_TOLERANCE].sort_values(["k", "aicc"]).iloc[0]
    table["selected"] = table.model == chosen.model
    table["delta_aicc"] = table.aicc - best
    return table.sort_values("aicc"), fits[chosen.model], chosen


def selected_outputs(key, title, axis, values, fit_info, chosen, steps, unit):
    fit, offset, scale = fit_info
    burn = int(fit.loglikelihood_burn)
    innovation = fit.filter_results.standardized_forecasts_error[0, burn:]
    fitted = np.asarray(fit.fittedvalues) * scale + offset
    fitted[:burn] = np.nan  # 积分模型的弥散初始化不作为有效样本内拟合。
    valid = np.arange(len(values)) >= burn
    fitted_table = pd.DataFrame({
        "time": axis, "actual": values, "one_step_fitted": fitted,
        "residual": values - fitted, "used_for_diagnostics": valid,
    })
    save_csv(fitted_table, f"{key}_fitted")
    lb = acorr_ljungbox(innovation, lags=RESIDUAL_LAGS,
                       model_df=int(chosen.p + chosen.q)).rename_axis("lag").reset_index()
    save_csv(lb, f"{key}_residual_ljung_box")
    pred = fit.get_forecast(steps=steps)
    ci = np.asarray(pred.conf_int(alpha=ALPHA)) * scale + offset
    mean = np.asarray(pred.predicted_mean) * scale + offset
    se = np.asarray(pred.se_mean) * scale
    future_axis = np.arange(axis[-1] + 1, axis[-1] + steps + 1)
    forecast = pd.DataFrame({"time": future_axis, "horizon": np.arange(1, steps + 1),
                             "forecast": mean, "standard_error": se,
                             "lower_95": ci[:, 0], "upper_95": ci[:, 1]})
    save_csv(forecast, f"{key}_forecast")
    ci_param = np.asarray(fit.conf_int())
    parameters = []
    for i, name in enumerate(fit.param_names):
        multiplier = scale ** 2 if name == "sigma2" else scale if name in ["const", "x1"] else 1
        shift = offset if name == "const" else 0
        parameters.append({
            "parameter": "drift" if name == "x1" else name,
            "estimate": float(fit.params[i] * multiplier + shift),
            "standard_error": float(fit.bse[i] * multiplier),
            # 常均值原假设转换至原尺度；漂移和动态系数的零假设不变。
            "p_value": float(2 * stats.norm.sf(abs((fit.params[i] * multiplier + shift) /
                                                   (fit.bse[i] * multiplier)))),
            "lower_95": float(ci_param[i, 0] * multiplier + shift),
            "upper_95": float(ci_param[i, 1] * multiplier + shift),
        })
    save_csv(pd.DataFrame(parameters), f"{key}_parameters")
    pd.DataFrame({"time": axis[burn:], "standardized_innovation": innovation}).to_csv(
        RESULTS / f"{key}_innovations.csv", index=False, float_format="%.12g")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.1))
    axes[0].plot(axis, values, color=BLUE, lw=1.2, label="观测值")
    axes[0].plot(axis, fitted, color=ORANGE, lw=1.1, alpha=0.85, label="一步预测拟合值")
    axes[0].set(title=title + "：样本内拟合", ylabel="原数据尺度")
    tail = max(0, len(values) - max(30, steps * 3))
    axes[1].plot(axis[tail:], values[tail:], color=BLUE, lw=1.3, label="历史观测")
    axes[1].plot(np.r_[axis[-1], future_axis], np.r_[values[-1], mean],
                 color=ORANGE, marker="o", ms=3, lw=1.3, label="点预测")
    if steps == 1:
        axes[1].errorbar(future_axis, mean, yerr=[mean - ci[:, 0], ci[:, 1] - mean],
                         fmt="none", color=ORANGE, capsize=6, lw=1.5,
                         label="条件95%预测区间")
    else:
        axes[1].fill_between(future_axis, ci[:, 0], ci[:, 1], color=ORANGE, alpha=0.18,
                             label="条件95%预测区间")
    axes[1].axvline(axis[-1] + 0.5, color="0.5", ls="--", lw=0.8)
    axes[1].set(title=f"{title}：未来{steps}{'次记录' if unit != '年' else '年'}预测",
                xlabel=unit, ylabel="原数据尺度")
    for ax in axes:
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    fig.suptitle(str(chosen.model), fontsize=12)
    save_figure(fig, f"{key}_fit_forecast")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    axes[0, 0].plot(axis[burn:], innovation, color=BLUE, lw=1)
    axes[0, 0].axhline(0, color="0.4", lw=0.7)
    axes[0, 0].set(title="标准化一步预测误差", xlabel=unit)
    axes[0, 1].hist(innovation, bins="auto", density=True, color=BLUE, alpha=0.7)
    grid = np.linspace(min(-3.5, innovation.min()), max(3.5, innovation.max()), 200)
    axes[0, 1].plot(grid, stats.norm.pdf(grid), color=ORANGE, label="标准正态密度")
    axes[0, 1].set(title="误差分布", xlabel="标准化误差")
    axes[0, 1].legend(fontsize=9)
    stats.probplot(innovation, dist="norm", plot=axes[1, 0])
    axes[1, 0].set(title="正态 Q-Q 图", xlabel="理论分位数", ylabel="样本分位数")
    ra = acf(innovation, nlags=min(20, len(innovation) // 2 - 1), fft=False)[1:]
    lag = np.arange(1, len(ra) + 1)
    axes[1, 1].vlines(lag, 0, ra, color=BLUE)
    axes[1, 1].plot(lag, ra, "o", ms=3, color=BLUE)
    bound = stats.norm.ppf(0.975) / np.sqrt(len(innovation))
    axes[1, 1].axhspan(-bound, bound, color="#dce8f4", alpha=0.8)
    axes[1, 1].axhline(0, color="0.4", lw=0.7)
    axes[1, 1].set(title=f"残差 ACF；LB(10) p={chosen.lb10_p:.3f}", xlabel="滞后阶数")
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    fig.suptitle(title + "：残差诊断", fontsize=12)
    save_figure(fig, f"{key}_residuals")
    jb = stats.jarque_bera(innovation)
    return forecast, {
        "parameters": parameters, "burn_in": burn,
        "ar_root_moduli": np.abs(fit.arroots).tolist(),
        "ma_root_moduli": np.abs(fit.maroots).tolist(),
        "in_sample_one_step_rmse": float(np.sqrt(np.mean((values[valid] - fitted[valid]) ** 2))),
        "jarque_bera_stat": float(jb.statistic), "jarque_bera_p": float(jb.pvalue),
        "forecast": forecast.to_dict(orient="records"),
    }


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    diagnostics, summaries, forecasts = [], {}, {}
    for key, source, column, title, expected_n, steps, unit in SPECS:
        frame = pd.read_csv(ROOT / "doc" / f"{source}.csv")
        assert len(frame) == expected_n and not frame.isna().any().any()
        axis, values = frame.iloc[:, 0].to_numpy(), frame[column].to_numpy(dtype=float)
        assert np.all(np.diff(axis) == 1)
        tests = [diagnostic(np.diff(values, n=d)) for d in range(3)]
        # 先选择同时拒绝单位根且不拒绝水平平稳的最小差分阶数。
        supported = [d for d, test in enumerate(tests)
                     if test["adf_p"] < ALPHA and test["kpss_p"] > ALPHA]
        if not supported:
            raise RuntimeError(f"{title}未找到受两种检验共同支持的差分阶数。")
        d = supported[0]
        for degree, test in enumerate(tests):
            diagnostics.append({"series": key, "difference": degree, "selected_d": d, **test})
        trend_test = diagnostic(values, regression="ct")
        summaries[key] = {"title": title, "source": source, "column": column,
                          "n": len(values), "start": int(axis[0]), "end": int(axis[-1]),
                          "d": d, "horizon": steps, "tests": tests,
                          "trend_stationarity_sensitivity": trend_test}
        fig, axes = plt.subplots(1 + bool(d), 1, figsize=(10, 3.2 + 2.6 * bool(d)), squeeze=False)
        axes[0, 0].plot(axis, values, color=BLUE, lw=1.2)
        axes[0, 0].set(title=title + "：原序列", xlabel=unit, ylabel="原数据尺度")
        if d:
            diff = np.diff(values, n=d)
            axes[1, 0].plot(axis[d:], diff, color=ORANGE, lw=1.1)
            axes[1, 0].axhline(0, color="0.5", lw=0.7)
            axes[1, 0].set(title=f"{d}阶普通差分", xlabel=unit, ylabel="差分值")
        for ax in axes.flat:
            ax.grid(alpha=0.2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
        save_figure(fig, f"{key}_time_difference")
        for degree in sorted({0, d}):
            suffix = "original" if degree == 0 else f"diff{degree}"
            corr = correlations(np.diff(values, n=degree))
            save_csv(corr, f"{key}_{suffix}_correlations")
            plot_correlations(corr, title + ("原序列" if degree == 0 else f"{degree}阶差分"),
                              f"{key}_{suffix}_acf_pacf")
        table, fit_info, chosen = fit_candidates(values, d)
        save_csv(table, f"{key}_model_comparison")
        forecast, details = selected_outputs(key, title, axis, values, fit_info, chosen, steps, unit)
        forecasts[key] = forecast
        selected = {name: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                           else value) for name, value in chosen.to_dict().items()}
        summaries[key].update({"selected": selected, "model_count": len(table),
                               "converged_count": int(table.converged.sum()),
                               "eligible_count": int(table.eligible.sum()), **details})
        print(f"{key}: d={d}, {chosen.model}, AICc={chosen.aicc:.4f}, "
              f"LB(6)={chosen.lb6_p:.4f}, LB(10)={chosen.lb10_p:.4f}", flush=True)
    save_csv(pd.DataFrame(diagnostics), "diagnostics")
    pop = pd.read_csv(ROOT / "doc" / "E5_4.csv")
    history_gap = pop.birth_rate - pop.mortality - pop.ngr
    reconciliation = pd.DataFrame({
        "year": forecasts["ngr"].time,
        "birth_forecast": forecasts["birth_rate"].forecast,
        "mortality_forecast": forecasts["mortality"].forecast,
        "ngr_independent_forecast": forecasts["ngr"].forecast,
        "birth_minus_mortality": forecasts["birth_rate"].forecast - forecasts["mortality"].forecast,
    })
    reconciliation["identity_gap"] = reconciliation.ngr_independent_forecast - reconciliation.birth_minus_mortality
    save_csv(reconciliation, "population_identity_comparison")
    pd.DataFrame({"year": pop.year, "birth_minus_mortality_minus_ngr": history_gap}).to_csv(
        RESULTS / "population_source_identity.csv", index=False, float_format="%.12g")
    overview = pd.DataFrame([{
        "series": key, "title": s["title"], "n": s["n"], "d": s["d"],
        "model": s["selected"]["model"], "aicc": s["selected"]["aicc"],
        "lb6_p": s["selected"]["lb6_p"], "lb10_p": s["selected"]["lb10_p"],
        "forecast_start": s["forecast"][0]["time"], "forecast_end": s["forecast"][-1]["time"],
        "first_forecast": s["forecast"][0]["forecast"],
        "last_forecast": s["forecast"][-1]["forecast"],
    } for key, s in summaries.items()])
    save_csv(overview, "model_summary")
    result = {
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "settings": {"alpha": ALPHA, "primary_adf_maxlag": ADF_MAXLAG,
                     "adf_autolag": "AIC", "kpss_nlags": "auto", "pacf_method": "ywm",
                     "candidate_p_q": "0..4 with p+q<=4", "aicc_tolerance": AICC_TOLERANCE,
                     "residual_lags": RESIDUAL_LAGS,
                     "selection": "converged and both residual LB p>0.05; minimum k within best AICc+2"},
        "population_source_max_identity_gap": float(np.abs(history_gap).max()),
        "series": summaries,
    }
    (RESULTS / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
