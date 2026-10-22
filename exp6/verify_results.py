"""独立核对落盘数据、中心移动平均、季节效应、回测及20季度预测，并记录文件摘要。"""
from pathlib import Path
import hashlib
import json

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(size, degree, seasonal, start=0):
    t = (np.arange(start, start + size) + 1) / 4
    columns = [t ** k for k in range(degree + 1)]
    if seasonal:
        columns.extend((np.arange(start, start + size) % 4 == q).astype(float) for q in [1, 2, 3])
    return np.column_stack(columns)


def main():
    summary = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
    audit = json.loads((RESULTS / "source_audit.json").read_text(encoding="utf-8"))
    for entry in audit:
        csv = ROOT / "doc" / entry["csv"]
        raw = ROOT / "doc" / entry["source"]
        assert digest(csv) == entry["csv_sha256"]
        if raw.exists():
            assert digest(raw) == entry["source_sha256"], "本地教学原件发生变化"
            original = pd.read_excel(raw)
            pd.testing.assert_frame_equal(original, pd.read_csv(csv, parse_dates=["time"]))
    for name, start, end, freq, count in [("E6_2", "1973-01-01", "1978-12-01", "MS", 72),
                                         ("E6_7", "1948-01-01", "1979-10-01", "QS-JAN", 128)]:
        data = pd.read_csv(ROOT / "doc" / f"{name}.csv", parse_dates=["time"])
        assert len(data) == count and list(data.columns) == ["time", "x"]
        assert not data.isna().any().any() and (data.x > 0).all()
        np.testing.assert_array_equal(data.time, pd.date_range(start, end, freq=freq))
    deaths = pd.read_csv(ROOT / "doc" / "E6_2.csv", parse_dates=["time"])
    y = deaths.x.to_numpy(dtype=float)
    # 独立按2×12公式计算第7至66个月趋势：端点权重1/24，中间11期权重1/12。
    trend = np.array([(y[t - 6] + y[t + 6]) / 24 + y[t - 5:t + 6].sum() / 12
                      for t in range(6, 66)])
    effects = pd.read_csv(RESULTS / "e6_2_seasonal_effects.csv")
    for method in ["additive", "multiplicative"]:
        out = pd.read_csv(RESULTS / f"e6_2_{method}_components.csv", parse_dates=["time"])
        np.testing.assert_array_equal(out.time, deaths.time)
        np.testing.assert_array_equal(out.observed, y)
        assert out.trend_available.sum() == 60
        assert out.trend.iloc[:6].isna().all() and out.trend.iloc[-6:].isna().all()
        assert out.irregular.iloc[:6].isna().all() and out.irregular.iloc[-6:].isna().all()
        np.testing.assert_allclose(out.trend.iloc[6:66], trend, atol=1e-7, rtol=0)
        detrended = y[6:66] - trend if method == "additive" else y[6:66] / trend
        months = deaths.time.dt.month.to_numpy()[6:66]
        independent = np.array([detrended[months == m].mean() for m in range(1, 13)])
        independent = independent - independent.mean() if method == "additive" else independent / independent.mean()
        column = "additive_effect" if method == "additive" else "multiplicative_index"
        np.testing.assert_allclose(effects[column], independent, atol=1e-8, rtol=0)
        np.testing.assert_allclose(out.seasonal, np.tile(independent, 6), atol=1e-8, rtol=0)
        reconstruction = out.trend + out.seasonal + out.irregular if method == "additive" else out.trend * out.seasonal * out.irregular
        np.testing.assert_allclose(reconstruction.iloc[6:66], y[6:66], atol=1e-7, rtol=0)
    np.testing.assert_allclose(effects.additive_effect.sum(), 0, atol=1e-8)
    np.testing.assert_allclose(effects.multiplicative_index.mean(), 1, atol=1e-10)
    income = pd.read_csv(ROOT / "doc" / "E6_7.csv", parse_dates=["time"])
    y = income.x.to_numpy(dtype=float)
    comparison = pd.read_csv(RESULTS / "e6_7_model_comparison.csv")
    backtest = pd.read_csv(RESULTS / "e6_7_backtest_predictions.csv", parse_dates=["time"])
    folds = pd.read_csv(RESULTS / "e6_7_backtest_folds.csv")
    assert len(comparison) == 15 and comparison.selected.sum() == 1
    assert len(backtest) == 15 * 11 * 20 and len(folds) == 15 * 11
    for _, row in comparison.iterrows():
        errors = backtest.loc[backtest.model == row.model, "error"].to_numpy()
        np.testing.assert_allclose(row.cv_rmse, np.sqrt(np.mean(errors ** 2)), atol=1e-7, rtol=0)
        for origin in range(68, 109, 4):
            fold = backtest.loc[(backtest.model == row.model) & (backtest.train_n == origin)]
            np.testing.assert_array_equal(fold.horizon, np.arange(1, 21))
            np.testing.assert_array_equal(fold.time, income.time.iloc[origin:origin + 20])
            np.testing.assert_array_equal(fold.actual, y[origin:origin + 20])
            train = y[:origin]
            if not row.is_baseline:
                degree, seasonal = int(row.degree), bool(row.seasonal)
                target = np.log(train) if row.scale == "log" else train
                x = matrix(origin, degree, seasonal)
                beta = np.linalg.lstsq(x, target, rcond=None)[0]
                prediction = matrix(20, degree, seasonal, origin) @ beta
                if row.scale == "log":
                    prediction = np.exp(prediction) * np.mean(np.exp(target - x @ beta))
            elif row.model == "last_value":
                prediction = np.repeat(train[-1], 20)
            elif row.model == "seasonal_naive":
                prediction = np.tile(train[-4:], 5)
            else:
                prediction = train[-1] + np.arange(1, 21) * (train[-1] - train[0]) / (origin - 1)
            np.testing.assert_allclose(fold.forecast, prediction, atol=2e-6, rtol=0)
            np.testing.assert_allclose(fold.error, prediction - y[origin:origin + 20], atol=2e-6, rtol=0)
        if not row.is_baseline:
            n, k = len(y), row.k
            expected = row.aic_original_scale + 2 * k * (k + 1) / (n - k - 1)
            np.testing.assert_allclose(row.aicc_original_scale, expected, atol=1e-7, rtol=0)
    candidates = comparison.loc[~comparison.is_baseline]
    pool = candidates.loc[candidates.cv_rmse <= candidates.cv_rmse.min() * 1.01]
    expected = pool.sort_values(["regression_parameters", "cv_rmse", "model"]).iloc[0]
    chosen = comparison.loc[comparison.selected].iloc[0]
    assert chosen.model == expected.model == summary["e6_7"]["selected_model"]
    params = pd.read_csv(RESULTS / "e6_7_parameters.csv")
    forecast = pd.read_csv(RESULTS / "e6_7_forecast.csv", parse_dates=["time"])
    fitted = pd.read_csv(RESULTS / "e6_7_fitted.csv", parse_dates=["time"])
    assert len(forecast) == 20 and len(fitted) == 128
    np.testing.assert_array_equal(forecast.time, pd.date_range("1980-01-01", "1984-10-01", freq="QS-JAN"))
    np.testing.assert_array_equal(forecast.quarter, forecast.time.dt.to_period("Q").astype(str))
    x_future = matrix(20, int(chosen.degree), bool(chosen.seasonal), 128)
    independent = x_future @ params.estimate.to_numpy()
    if chosen.scale == "log":
        independent = np.exp(independent) * np.mean(np.exp(fitted.residual_model_scale))
    np.testing.assert_allclose(forecast.forecast, independent, atol=2e-6, rtol=0)
    assert np.isfinite(forecast.forecast).all() and (forecast.forecast > 0).all()
    alternatives = pd.read_csv(RESULTS / "e6_7_candidate_forecasts.csv")
    assert len(alternatives) == 12 * 20 and alternatives.selected.sum() == 20
    for _, row in candidates.iterrows():
        target = np.log(y) if row.scale == "log" else y
        x = matrix(len(y), int(row.degree), bool(row.seasonal))
        beta = np.linalg.lstsq(x, target, rcond=None)[0]
        alt_prediction = matrix(20, int(row.degree), bool(row.seasonal), 128) @ beta
        if row.scale == "log":
            alt_prediction = np.exp(alt_prediction) * np.mean(np.exp(target - x @ beta))
        np.testing.assert_allclose(alternatives.loc[alternatives.model == row.model, "forecast"],
                                   alt_prediction, atol=2e-6, rtol=0)
    np.testing.assert_allclose(alternatives.loc[alternatives.selected, "forecast"], forecast.forecast,
                               atol=1e-7, rtol=0)
    annual = pd.read_csv(RESULTS / "e6_7_forecast_annual_average.csv")
    np.testing.assert_array_equal(annual.year, np.arange(1980, 1985))
    assert (annual.quarters == 4).all()
    np.testing.assert_allclose(annual.quarterly_forecast_average,
                               forecast.forecast.to_numpy().reshape(5, 4).mean(axis=1), atol=1e-7, rtol=0)
    figures = list((ROOT / "figure").glob("*.png"))
    assert len(figures) == 10
    for path in figures:
        image = mpimg.imread(path)
        assert image.shape[0] >= 700 and image.shape[1] >= 1800 and image.std() > 0.01
    verification = {"passed": True, "source_rows": 200, "decomposition_methods": 2,
                    "trend_values_per_method": 60, "candidate_models": 12, "baselines": 3,
                    "backtest_folds": 165, "backtest_predictions": 3300, "forecast_quarters": 20,
                    "png_files": len(figures),
                    "checks": ["CSV哈希与本地原件逐值一致（原件存在时）", "日期范围和频率", "2×12独立趋势递推",
                               "独立季节效应及加法/乘法重构", "3300个回测预测独立最小二乘复算",
                               "模型选择规则与AICc", "参数重算20季度预测与5个年度季度均值", "全部PNG可读取"]}
    (RESULTS / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and not any(part in {"qa", "report", "__pycache__"} for part in p.relative_to(ROOT).parts)
             and p.suffix.lower() not in {".xlsx", ".doc", ".docx", ".pdf", ".pyc"}
             and p.name != "artifact_manifest.csv"]
    manifest = pd.DataFrame({"path": [p.relative_to(ROOT).as_posix() for p in sorted(files)],
                             "bytes": [p.stat().st_size for p in sorted(files)],
                             "sha256": [digest(p) for p in sorted(files)]})
    manifest.to_csv(RESULTS / "artifact_manifest.csv", index=False, lineterminator="\n")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
