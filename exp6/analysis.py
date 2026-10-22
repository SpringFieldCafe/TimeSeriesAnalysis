"""实验六：事故死亡数据的确定性分解、季度收入趋势拟合及五年预测。"""
from pathlib import Path
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf


ROOT = Path(__file__).resolve().parent
RESULTS, FIGURE = ROOT / "results", ROOT / "figure"
BLUE, ORANGE = "#245b8f", "#b96128"
HORIZON, ORIGINS, RMSE_TOLERANCE = 20, list(range(68, 109, 4)), 0.01
MODELS = [(scale, degree, seasonal) for scale in ["level", "log"]
          for degree in [1, 2, 3] for seasonal in [False, True]]


def save_csv(frame, name):
    frame.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.12g",
                 date_format="%Y-%m-%d", lineterminator="\n")


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


def model_name(scale, degree, seasonal):
    return f"{scale}_degree{degree}" + ("_quarter" if seasonal else "")


def design(n, degree, seasonal, start=0):
    # 固定原点：1948Q1 t=0.25，每个季度递增0.25年；训练和预测共享同一原点。
    t = (np.arange(start, start + n) + 1) / 4.0
    data = {"const": np.ones(n)}
    data.update({f"t_pow{i}": t ** i for i in range(1, degree + 1)})
    if seasonal:
        data.update({f"Q{q}": (np.arange(start, start + n) % 4 == q - 1).astype(float)
                     for q in [2, 3, 4]})
    return pd.DataFrame(data)


def fit_model(values, spec):
    scale, degree, seasonal = spec
    target = np.log(values) if scale == "log" else values
    return sm.OLS(target, design(len(values), degree, seasonal)).fit()


def predict_model(fit, spec, n, start):
    scale, degree, seasonal = spec
    linear = fit.predict(design(n, degree, seasonal, start)).to_numpy()
    # 对数模型以训练残差的 Duan smearing 修正重变换偏差；每折仅用训练样本。
    smearing = float(np.mean(np.exp(fit.resid))) if scale == "log" else 1.0
    median = np.exp(linear) if scale == "log" else linear
    return median * smearing, median, linear, smearing


def residual_diagnostics(values, dates, name, period):
    residual = np.asarray(values, dtype=float)
    lag_count = min(24, len(residual) // 2 - 1)
    corr, ci = acf(residual, nlags=lag_count, alpha=0.05, fft=False)
    save_csv(pd.DataFrame({"lag": np.arange(1, lag_count + 1), "acf": corr[1:],
                          "ci_lower": ci[1:, 0], "ci_upper": ci[1:, 1]}), f"{name}_residual_acf")
    lags = sorted(set([period, 2 * period]))
    lb = acorr_ljungbox(residual, lags=lags, return_df=True).rename_axis("lag").reset_index()
    save_csv(lb, f"{name}_residual_ljung_box")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(dates, residual, color=BLUE, lw=1.2)
    axes[0, 0].axhline(0, color="0.4", lw=0.8)
    axes[0, 0].set(title="余项随时间变化", ylabel="建模尺度上的余项")
    axes[0, 1].vlines(np.arange(1, lag_count + 1), 0, corr[1:], color=BLUE)
    axes[0, 1].fill_between(np.arange(1, lag_count + 1), ci[1:, 0] - corr[1:],
                            ci[1:, 1] - corr[1:], color="#dce8f4")
    axes[0, 1].axhline(0, color="0.4", lw=0.8)
    axes[0, 1].set(title="余项 ACF（95%逐阶参考区间）", xlabel="滞后阶数")
    axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1, 0].hist(residual, bins=12, color=BLUE, alpha=0.8)
    axes[1, 0].set(title="余项分布", xlabel="建模尺度上的余项")
    stats.probplot(residual, dist="norm", plot=axes[1, 1])
    axes[1, 1].set(title="正态 Q-Q 图", xlabel="理论分位数", ylabel="样本分位数")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    save_figure(fig, f"{name}_residuals")
    jb = stats.jarque_bera(residual)
    return {"n": len(residual), "jarque_bera_p": float(jb.pvalue),
            "ljung_box": lb.to_dict(orient="records")}


def accidents():
    frame = pd.read_csv(ROOT / "doc" / "E6_2.csv", parse_dates=["time"])
    y = frame.set_index("time").x.asfreq("MS")
    assert len(y) == 72 and y.notna().all()
    # 偶数周期使用2×12中心化移动平均。两端各6期无趋势估计，不外推填补。
    decompositions, comparison = {}, []
    for method in ["additive", "multiplicative"]:
        decomposition = seasonal_decompose(y, model=method, period=12,
                                           two_sided=True, extrapolate_trend=0)
        trend, season = decomposition.trend, decomposition.seasonal
        fitted = trend + season if method == "additive" else trend * season
        out = pd.DataFrame({"time": y.index, "observed": y.to_numpy(), "trend": trend.to_numpy(),
                            "seasonal": season.to_numpy(), "fitted": fitted.to_numpy(),
                            "irregular": decomposition.resid.to_numpy(),
                            "seasonally_adjusted": (y - season if method == "additive" else y / season).to_numpy(),
                            "trend_available": trend.notna().to_numpy()})
        save_csv(out, f"e6_2_{method}_components")
        decompositions[method] = out
        mask = out.trend_available
        error = out.loc[mask, "observed"] - out.loc[mask, "fitted"]
        comparison.append({"model": method, "n_common": int(mask.sum()),
                           "original_scale_rmse": float(np.sqrt(np.mean(error ** 2))),
                           "original_scale_mae": float(np.mean(np.abs(error))),
                           "primary": method == "additive"})
    save_csv(pd.DataFrame(comparison), "e6_2_decomposition_comparison")
    main, mult = decompositions["additive"], decompositions["multiplicative"]
    monthly = pd.DataFrame({"month": np.arange(1, 13), "additive_effect": main.seasonal.iloc[:12].to_numpy(),
                            "multiplicative_index": mult.seasonal.iloc[:12].to_numpy()})
    save_csv(monthly, "e6_2_seasonal_effects")
    annual = frame.assign(year=frame.time.dt.year).groupby("year").x.agg(["mean", "min", "max"])
    annual["range"] = annual["max"] - annual["min"]
    annual["relative_range"] = annual["range"] / annual["mean"]
    save_csv(annual.reset_index(), "e6_2_annual_summary")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(frame.time, frame.x, color=BLUE, lw=1.3)
    axes[0].set(title="1973—1978年美国月度事故死亡数据", ylabel="死亡人数（原始月度总数）")
    for year, group in frame.groupby(frame.time.dt.year):
        axes[1].plot(group.time.dt.month, group.x, marker="o", ms=3, label=str(year))
    axes[1].set(title="各年按月份对齐", xlabel="月份", ylabel="死亡人数", xticks=np.arange(1, 13))
    axes[1].legend(ncol=3, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    save_figure(fig, "e6_2_time_seasonal")
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, column, title in zip(axes, ["observed", "trend", "seasonal", "irregular"],
                                 ["原始序列", "趋势与低频变化（两端各6期不估计）", "季节效应（每12期重复）", "不规则余项"]):
        ax.plot(main.time, main[column], color=BLUE, lw=1.3)
        ax.set(ylabel=title)
        ax.grid(alpha=0.2)
    axes[0].set_title("加法分解：观测 = 趋势 + 季节 + 余项")
    save_figure(fig, "e6_2_decomposition")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly.month, monthly.additive_effect, color=BLUE)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(title="月度季节效应（12个月效应之和为0）", xlabel="月份", ylabel="相对趋势的增减人数",
           xticks=np.arange(1, 13))
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "e6_2_seasonal_effects")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(main.time, main.observed, color="0.65", label="原始数据")
    ax.plot(main.time, main.seasonally_adjusted, color=BLUE, label="去季节后的数据")
    ax.plot(main.time, main.trend, color=ORANGE, lw=2, label="中心移动平均趋势")
    ax.set(title="去季节与趋势提取", ylabel="死亡人数")
    ax.legend(); ax.grid(alpha=0.2)
    save_figure(fig, "e6_2_trend_adjusted")
    valid = main.trend_available
    diagnostic = residual_diagnostics(main.loc[valid, "irregular"], main.loc[valid, "time"], "e6_2", 12)
    return {"n": len(y), "period": 12, "primary": "additive", "available_trends": int(valid.sum()),
            "trend_min_date": main.loc[main.trend.idxmin(), "time"].strftime("%Y-%m-%d"),
            "trend_min": float(main.trend.min()), "trend_start": float(main.trend.dropna().iloc[0]),
            "trend_end": float(main.trend.dropna().iloc[-1]),
            "seasonal_peak_month": int(monthly.loc[monthly.additive_effect.idxmax(), "month"]),
            "seasonal_low_month": int(monthly.loc[monthly.additive_effect.idxmin(), "month"]),
            "diagnostics": diagnostic, "comparison": comparison}


def income():
    frame = pd.read_csv(ROOT / "doc" / "E6_7.csv", parse_dates=["time"])
    y, n = frame.x.to_numpy(dtype=float), len(frame)
    assert n == 128 and frame.time.iloc[-1] == pd.Timestamp("1979-10-01")
    specs = {model_name(*spec): spec for spec in MODELS}
    folds, predictions, comparison, fits = [], [], [], {}
    for name in [*specs, "last_value", "seasonal_naive", "drift"]:
        for origin in ORIGINS:
            train = y[:origin]
            if name in specs:
                fit = fit_model(train, specs[name])
                forecast, _, _, _ = predict_model(fit, specs[name], HORIZON, origin)
            elif name == "last_value":
                forecast = np.repeat(train[-1], HORIZON)
            elif name == "seasonal_naive":
                forecast = np.tile(train[-4:], HORIZON // 4)
            else:
                forecast = train[-1] + np.arange(1, HORIZON + 1) * (train[-1] - train[0]) / (origin - 1)
            actual = y[origin:origin + HORIZON]
            error = forecast - actual
            folds.append({"model": name, "train_n": origin,
                          "train_end": str(frame.time.iloc[origin - 1].date()),
                          "test_end": str(frame.time.iloc[origin + HORIZON - 1].date()),
                          "rmse": float(np.sqrt(np.mean(error ** 2))),
                          "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))})
            predictions.extend({"model": name, "train_n": origin, "horizon": h + 1,
                                "time": frame.time.iloc[origin + h], "actual": actual[h],
                                "forecast": forecast[h], "error": error[h]} for h in range(HORIZON))
    backtest = pd.DataFrame(predictions)
    save_csv(pd.DataFrame(folds), "e6_7_backtest_folds")
    save_csv(backtest, "e6_7_backtest_predictions")
    for name in [*specs, "last_value", "seasonal_naive", "drift"]:
        errors = backtest.loc[backtest.model == name, "error"].to_numpy()
        row = {"model": name, "is_baseline": name not in specs,
               "cv_rmse": float(np.sqrt(np.mean(errors ** 2))),
               "cv_mae": float(np.mean(np.abs(errors))), "cv_bias": float(np.mean(errors)),
               "folds": len(ORIGINS), "forecast_pairs": len(errors)}
        if name in specs:
            spec = specs[name]
            fit = fit_model(y, spec); fits[name] = fit
            fitted, _, _, _ = predict_model(fit, spec, n, 0)
            # 对数模型补雅可比，使所有AICc对应原始观测尺度；计入创新方差参数。
            k = len(fit.params) + 1
            loglike = float(fit.llf - np.log(y).sum()) if spec[0] == "log" else float(fit.llf)
            aic = -2 * loglike + 2 * k
            row.update({"scale": spec[0], "degree": spec[1], "seasonal": spec[2],
                        "regression_parameters": len(fit.params), "k": k, "aic_original_scale": aic,
                        "aicc_original_scale": aic + 2 * k * (k + 1) / (n - k - 1),
                        "in_sample_rmse": float(np.sqrt(np.mean((y - fitted) ** 2)))})
        comparison.append(row)
    table = pd.DataFrame(comparison)
    candidates = table.loc[~table.is_baseline]
    threshold = float(candidates.cv_rmse.min() * (1 + RMSE_TOLERANCE))
    pool = candidates.loc[candidates.cv_rmse <= threshold]
    chosen = pool.sort_values(["regression_parameters", "cv_rmse", "model"]).iloc[0]
    name, spec = str(chosen.model), specs[str(chosen.model)]
    table["selected"] = table.model == name
    save_csv(table.sort_values("cv_rmse"), "e6_7_model_comparison")
    fit = fits[name]
    fitted, median, linear, smearing = predict_model(fit, spec, n, 0)
    forecast, forecast_median, future_linear, _ = predict_model(fit, spec, HORIZON, n)
    future = pd.date_range("1980-01-01", periods=HORIZON, freq="QS-JAN")
    save_csv(pd.DataFrame({"time": frame.time, "quarter": frame.time.dt.to_period("Q").astype(str),
                          "observed": y, "fitted": fitted, "trend_scale_fit": linear,
                          "unadjusted_backtransform": median, "residual_original_scale": y - fitted,
                          "residual_model_scale": fit.resid}), "e6_7_fitted")
    fc = pd.DataFrame({"time": future, "quarter": future.to_period("Q").astype(str),
                       "horizon": np.arange(1, 21), "forecast": forecast,
                       "unadjusted_backtransform": forecast_median, "trend_scale_forecast": future_linear})
    save_csv(fc, "e6_7_forecast")
    # 人均收入为季度指标，年度列只保存四个季度的算术平均，不相加为年度人均收入。
    save_csv(fc.assign(year=future.year).groupby("year").agg(
        quarterly_forecast_average=("forecast", "mean"), quarters=("forecast", "size")).reset_index(),
        "e6_7_forecast_annual_average")
    alternatives = []
    for candidate_name, candidate_spec in specs.items():
        candidate_prediction, _, _, _ = predict_model(fits[candidate_name], candidate_spec, HORIZON, n)
        alternatives.extend({"model": candidate_name, "quarter": str(date.to_period("Q")),
                             "horizon": h + 1, "forecast": candidate_prediction[h],
                             "selected": candidate_name == name} for h, date in enumerate(future))
    save_csv(pd.DataFrame(alternatives), "e6_7_candidate_forecasts")
    hac = fit.get_robustcov_results(cov_type="HAC", maxlags=8)
    ci = hac.conf_int()
    params = pd.DataFrame({"parameter": fit.params.index, "estimate": fit.params.to_numpy(),
                           "ols_standard_error": fit.bse.to_numpy(), "hac8_standard_error": hac.bse,
                           "hac8_pvalue": hac.pvalues, "hac8_ci_lower": ci[:, 0], "hac8_ci_upper": ci[:, 1]})
    save_csv(params, "e6_7_parameters")
    # 在所选趋势阶数与尺度下，单独检验季度哑变量；HAC仅修正系数推断，不消除残差相关。
    seasonal_fit = fits[model_name(spec[0], spec[1], True)]
    seasonal_hac = seasonal_fit.get_robustcov_results(cov_type="HAC", maxlags=8)
    r = np.zeros((3, len(seasonal_fit.params))); r[:, -3:] = np.eye(3)
    seasonal_test = seasonal_hac.wald_test(r, use_f=True, scalar=True)
    season_table = pd.DataFrame({"quarter": [1, 2, 3, 4],
                                "effect_vs_Q1_model_scale": [0.0, *seasonal_fit.params.iloc[-3:]]})
    save_csv(season_table, "e6_7_quarter_effects")
    diagnostic = residual_diagnostics(fit.resid, frame.time, "e6_7", 4)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(frame.time, y, color=BLUE)
    axes[0].set(title="1948—1979年艾奥瓦州非农产品季度人均收入", ylabel="给定收入尺度")
    axes[1].plot(frame.time, np.log(y), color=BLUE)
    axes[1].set(title="对数尺度：检查增长速度是否恒定", ylabel="自然对数")
    for ax in axes:
        ax.grid(alpha=0.2)
    save_figure(fig, "e6_7_time")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame.time, y, color="0.65", label="实际")
    axes[0].plot(frame.time, fitted, color=BLUE, label="所选趋势拟合")
    axes[0].set(title="样本内拟合", ylabel="给定收入尺度"); axes[0].legend()
    axes[1].bar(season_table.quarter, season_table.effect_vs_Q1_model_scale, color=BLUE)
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set(title="同阶趋势控制后的季度效应（Q1为基准）", xlabel="季度",
                ylabel=f"{spec[0]} 尺度的相对效应", xticks=[1, 2, 3, 4])
    for ax in axes:
        ax.grid(alpha=0.2)
    save_figure(fig, "e6_7_fit_quarter_effects")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for ax, start in zip(axes, [0, 100]):
        ax.plot(frame.time.iloc[start:], y[start:], color="0.55", label="实际数据")
        ax.plot(frame.time.iloc[start:], fitted[start:], color=BLUE, label="样本内拟合")
        ax.plot(pd.DatetimeIndex([frame.time.iloc[-1], *future]), np.r_[fitted[-1], forecast],
                color=ORANGE, lw=1.8, label="1980—1984年点预测")
        ax.axvline(frame.time.iloc[-1], color="0.5", ls="--", lw=0.8)
        ax.set(ylabel="给定收入尺度"); ax.grid(alpha=0.2)
    axes[0].set_title(f"季度收入拟合与未来5年预测：{name}")
    axes[0].legend()
    axes[1].set_title("最近7年数据与未来5年点预测")
    save_figure(fig, "e6_7_forecast")
    fig, ax = plt.subplots(figsize=(11, 6))
    sorted_table = table.sort_values("cv_rmse", ascending=False)
    colors = [ORANGE if selected else BLUE for selected in sorted_table.selected]
    ax.barh(sorted_table.model, sorted_table.cv_rmse, color=colors)
    ax.set(title="11个滚动起点、每次20季度预测的历史回测", xlabel="原始收入尺度 RMSE（越小越好）")
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "e6_7_model_comparison")
    horizon_errors = backtest.loc[backtest.model == name].groupby("horizon").error.agg(
        rmse=lambda v: np.sqrt(np.mean(v ** 2)), mae=lambda v: np.mean(np.abs(v)), bias="mean").reset_index()
    save_csv(horizon_errors, "e6_7_backtest_by_horizon")
    return {"n": n, "period": 4, "horizon": HORIZON, "selected_model": name,
            "selection_rmse_threshold": threshold, "smearing_factor": smearing,
            "parameters": dict(zip(fit.params.index, map(float, fit.params))),
            "cv_rmse": float(chosen.cv_rmse), "in_sample_rmse": float(chosen.in_sample_rmse),
            "quarter_hac8_f": float(seasonal_test.statistic), "quarter_hac8_p": float(seasonal_test.pvalue),
            "forecast_first": float(forecast[0]), "forecast_last": float(forecast[-1]),
            "diagnostics": diagnostic}


def main():
    FIGURE.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    summary = {"versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
               "settings": {"income_horizon": HORIZON, "cv_train_sizes": ORIGINS,
                            "cv_horizon": HORIZON, "rmse_tolerance": RMSE_TOLERANCE,
                            "hac_maxlags": 8}, "e6_2": accidents(), "e6_7": income()}
    (RESULTS / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                    allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
