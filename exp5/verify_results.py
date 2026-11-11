"""核对实际输出：时间范围、AICc、区间及用模型方程独立递推的预测值。"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent


def main():
    results = ROOT / "results"
    content = json.loads((results / "results.json").read_text(encoding="utf-8"))
    checked = 0
    for key, entry in content["series"].items():
        raw = pd.read_csv(ROOT / "doc" / f"{entry['source']}.csv")
        y = raw[entry["column"]].to_numpy()
        forecast = pd.read_csv(results / f"{key}_forecast.csv")
        comparison = pd.read_csv(results / f"{key}_model_comparison.csv")
        chosen = comparison.loc[comparison.selected].iloc[0]
        params = pd.read_csv(results / f"{key}_parameters.csv").set_index("parameter").estimate
        fitted = pd.read_csv(results / f"{key}_fitted.csv")
        residual = pd.read_csv(results / f"{key}_residual_ljung_box.csv")
        assert comparison.selected.sum() == 1 and bool(chosen.eligible)
        assert chosen.converged and np.all(residual.lb_pvalue > 0.05)
        assert np.all(comparison.p + comparison.q <= 4)
        eligible = comparison.loc[comparison.eligible]
        pool = eligible.loc[eligible.aicc <= eligible.aicc.min() + 2].sort_values(["k", "aicc"])
        assert chosen.model == pool.iloc[0].model
        correction = 2 * chosen.k * (chosen.k + 1) / (chosen.n_effective - chosen.k - 1)
        np.testing.assert_allclose(chosen.aicc, chosen.aic + correction, rtol=0, atol=2e-8)
        np.testing.assert_array_equal(forecast.time, np.arange(entry["end"] + 1,
                                                               entry["end"] + entry["horizon"] + 1))
        assert np.isfinite(forecast.to_numpy()).all() and (forecast.standard_error > 0).all()
        for column, sign in [("lower_95", -1), ("upper_95", 1)]:
            np.testing.assert_allclose(forecast[column], forecast.forecast +
                                       sign * norm.ppf(0.975) * forecast.standard_error,
                                       rtol=0, atol=2e-8)
        p, d, q = int(chosen.p), int(chosen.d), int(chosen.q)
        mean = float(params.get("const", 0)) if d == 0 else float(params.get("drift", 0))
        past = list(np.diff(y, n=d))
        errors = list(fitted.residual)
        independent, level = [], y[-1]
        for _ in range(entry["horizon"]):
            next_value = mean + sum(params[f"ar.L{i}"] * (past[-i] - mean) for i in range(1, p + 1))
            next_value += sum(params[f"ma.L{i}"] * errors[-i] for i in range(1, q + 1))
            past.append(next_value)
            errors.append(0.0)  # 多步预测中未来创新的条件期望为0。
            level = level + next_value if d == 1 else next_value
            independent.append(level)
        np.testing.assert_allclose(forecast.forecast, independent, rtol=0, atol=2e-5)
        if p:
            roots = np.roots([-params[f"ar.L{i}"] for i in range(p, 0, -1)] + [1.0])
            assert (np.abs(roots) > 1).all()
        if q:
            roots = np.roots([params[f"ma.L{i}"] for i in range(q, 0, -1)] + [1.0])
            assert (np.abs(roots) > 1).all()
        figure_names = [f"{key}_time_difference", f"{key}_original_acf_pacf",
                        f"{key}_fit_forecast", f"{key}_residuals"]
        if d:
            figure_names.append(f"{key}_diff{d}_acf_pacf")
        for name in figure_names:
            assert (ROOT / "figure" / f"{name}.png").stat().st_size > 10000
        checked += 1
    print(f"Verified {checked} series: model selection, AICc, forecast recursion, intervals, roots and saved figures.")


if __name__ == "__main__":
    main()
