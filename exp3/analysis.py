"""实验三：时序图、平稳性及样本 ACF/PACF 特征。"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
OUTPUT = ROOT / "results"
LAGS = 24
ALPHA = 0.05
DATASETS = (
    ("E2_7", "mortality", "澳大利亚枪支相关凶杀案死亡率", "死亡率（每10万人）", 1915, 2004),
    ("E2_8", "wl", "密歇根湖每年月平均水位最高值", "水位（原数据单位）", 1860, 1955),
)
CONCLUSIONS = {
    "e2_7": {
        "stationarity": "时序图表现为多年尺度的起伏，后段明显下降，没有稳定的全期线性趋势；均值稳定性仍存疑。ADF 未拒绝单位根原假设，KPSS 未拒绝水平平稳原假设，两者没有给出一致证据。课堂图形分析可将其作为近似平稳的有相关性序列讨论，但不能据此确认平稳，更不能确认严格平稳。",
        "correlation": "ACF 随滞后增加总体衰减，并由正相关逐渐转为弱负相关，呈拖尾形态。PACF 前两阶较突出，第二阶仅略超出95%参考区间，第三阶以后大多接近零；第12阶存在孤立越界。因此可描述为近似二阶截尾，但二阶判断较弱，不宜直接确定 AR(2) 模型。",
    },
    "e2_8": {
        "stationarity": "时序图前期水位较高、后期较低，并伴随多年起伏，均值随时间改变。ACF 长时间保持较高正相关并缓慢衰减。ADF 在5%水平下未拒绝单位根原假设，KPSS 拒绝水平平稳原假设；综合判断原序列按非平稳序列处理。",
        "correlation": "ACF 缓慢衰减，呈明显拖尾形态。PACF 第一阶很大，第二阶负相关仅略超出95%参考区间，之后大多较小，第17、21阶有孤立越界。图形上可近似描述为二阶截尾（第二阶在临界附近），但原序列非平稳，不能据此套用平稳 AR(2) 的识别规则。",
    },
    "e2_8_diff": {
        "stationarity": "一阶差分 Δx_t = x_t − x_(t−1) 后，序列围绕接近零的均值波动，ADF 拒绝单位根原假设，KPSS 未拒绝水平平稳原假设，支持差分序列近似平稳。这是补充验证，并不证明原序列一定含单位根；本实验不进一步拟合或确定 ARIMA 阶数。",
        "correlation": "差分后的 ACF/PACF 不再呈原序列那样的长期高正相关，低阶仍有相关性；有限样本下高阶有零星越界，不能据此认定精确截尾阶数。",
    },
}


def read_series(filename, column, start, end):
    frame = pd.read_excel(ROOT / "doc" / f"{filename}.xlsx", sheet_name="Sheet1")
    frame = frame[["year", column]].sort_values("year").reset_index(drop=True)
    if frame.isna().any().any() or not np.array_equal(frame["year"], np.arange(start, end + 1)):
        raise ValueError(f"{filename}: 年份不连续或包含缺失值")
    frame["year"] = frame["year"].astype(int)
    frame[column] = frame[column].astype(float)
    return frame


def diagnostics(values):
    adf = adfuller(values, regression="c", autolag="AIC")
    # KPSS 返回的 0.01/0.10 是查表边界，输出中保留不等号。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kp = kpss(values, regression="c", nlags="auto")
    bound = ">" if kp[1] == 0.1 else "<" if kp[1] == 0.01 else "="
    return {
        "n": len(values), "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "adf_stat": float(adf[0]), "adf_p": float(adf[1]),
        "adf_lags": int(adf[2]), "adf_nobs": int(adf[3]),
        "adf_critical_values": {k: float(v) for k, v in adf[4].items()},
        "kpss_stat": float(kp[0]), "kpss_p": float(kp[1]),
        "kpss_p_bound": bound, "kpss_lags": int(kp[2]),
        "kpss_critical_values": {k: float(v) for k, v in kp[3].items()},
        "pacf_reference_halfwidth": float(norm.ppf(1 - ALPHA / 2) / np.sqrt(len(values))),
    }


def correlations(values):
    a, a_ci = acf(values, nlags=LAGS, alpha=ALPHA, fft=False, adjusted=False,
                  bartlett_confint=True)
    p, p_ci = pacf(values, nlags=LAGS, alpha=ALPHA, method="ywm")
    return pd.DataFrame({
        "lag": np.arange(LAGS + 1), "acf": a,
        "acf_ci_lower": a_ci[:, 0], "acf_ci_upper": a_ci[:, 1],
        "pacf": p, "pacf_ci_lower": p_ci[:, 0], "pacf_ci_upper": p_ci[:, 1],
        "acf_significant": (a_ci[:, 0] > 0) | (a_ci[:, 1] < 0),
        "pacf_significant": (p_ci[:, 0] > 0) | (p_ci[:, 1] < 0),
    })


def save_plots(years, values, title, ylabel, prefix, corr):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, values, color="#245b8f", lw=1.4, label="年度观测值")
    rolling = pd.Series(values).rolling(10, center=True).mean()
    ax.plot(years, rolling, color="#ba6429", lw=2, label="10年居中移动平均")
    ax.set(title=title, xlabel="年份", ylabel=ylabel)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{prefix}_time.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in zip(axes, ("acf", "pacf"), ("样本自相关 ACF", "样本偏自相关 PACF")):
        # 系数的置信区间中心在估计值；图中将区间平移到零轴。
        lags = corr["lag"].iloc[1:].to_numpy()
        coeff = corr[key].iloc[1:].to_numpy()
        lower = corr[f"{key}_ci_lower"].iloc[1:].to_numpy() - coeff
        upper = corr[f"{key}_ci_upper"].iloc[1:].to_numpy() - coeff
        ax.fill_between(lags, lower, upper, color="#dce8f4", label="95%逐阶参考区间")
        ax.vlines(lags, 0, coeff, color="#245b8f", lw=1.4)
        ax.plot(lags, coeff, "o", color="#245b8f", ms=3.6)
        ax.axhline(0, color="#606060", lw=0.8)
        ax.set(title=label, xlabel="滞后阶数（年）", ylabel="相关系数", xlim=(0.4, LAGS + 0.6), ylim=(-1.05, 1.05))
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{prefix}_acf_pacf.png", dpi=180)
    plt.close(fig)


def analyze(years, values, title, ylabel, prefix):
    corr = correlations(values)
    corr.to_csv(OUTPUT / f"{prefix}_correlations.csv", index=False, float_format="%.10g")
    save_plots(years, values, title, ylabel, prefix, corr)
    result = diagnostics(values)
    result.update({"title": title, "start_year": int(years[0]), "end_year": int(years[-1]),
                   "acf_significant_lags": corr.loc[(corr.lag > 0) & corr.acf_significant, "lag"].tolist(),
                   "pacf_significant_lags": corr.loc[(corr.lag > 0) & corr.pacf_significant, "lag"].tolist(),
                   "correlations": corr.to_dict(orient="records"), **CONCLUSIONS[prefix]})
    return result


def p_text(result):
    return f"{result['kpss_p_bound']}{result['kpss_p']:.2f}"


def write_readme(results):
    a, b, diff = (results[k] for k in ("e2_7", "e2_8", "e2_8_diff"))
    text = f"""# 实验三：平稳性与 ACF/PACF 的截尾、拖尾特征

依据本地题目，分别分析 1915–2004 年澳大利亚枪支相关凶杀案死亡率与 1860–1955 年密歇根湖每年月平均水位的最高值。后者是每年一个观测值，不是月度序列；原表未注明水位单位，因此保留原数据单位。

## 运行

在仓库根目录的 Anaconda Prompt 或已激活 Anaconda 的终端执行：

```powershell
conda activate base
python exp3/analysis.py
```

本机已有依赖，无需配置新环境。其他机器若缺包，优先在已有 Anaconda 环境中使用南京大学镜像：

```powershell
python -m pip install -r exp3/requirements.txt -i https://mirror.nju.edu.cn/pypi/web/simple
```

代码路径基于脚本目录，因此也可在 `exp3/` 内运行 `python analysis.py`。分析沿用实验二的 pandas、Matplotlib 和 statsmodels 方法，补充 PACF 与数值结果；不依赖 `exp2/` 文件。

## 方法与判读

原始数据位于 `doc/E2_7.xlsx`、`doc/E2_8.xlsx`，分别有 {a['n']} 和 {b['n']} 条连续年度记录，无缺失值。时序图同时显示原始观测与10年居中移动平均，用于观察局部均值变化；移动平均不参与检验。样本 ACF/PACF 展示1–24阶；ACF 使用分母 n 的样本自协方差与 Bartlett 95%逐阶参考区间，PACF 使用不作样本量修正的 Yule–Walker 估计（`ywm`），参考区间约为 ±1.96/√n。

ADF 原假设为存在单位根，采用常数项、AIC 选滞后阶数；KPSS 原假设为水平平稳，采用常数项、自动带宽。显著性水平为5%。KPSS 查表范围为0.01–0.10，边界结果以不等号表示。未拒绝原假设不能等同于证明原假设成立。

理论上，平稳 AR(p) 的 ACF 拖尾、PACF 在 p 阶后截尾；平稳 MA(q) 的 ACF 在 q 阶后截尾、PACF 拖尾。样本图只能判断近似形态，逐阶95%区间不作多重比较校正，零星越界不能单独否定近似截尾。对非平稳原序列，区间仅作图形参考，不能直接使用上述规则确定模型阶数。

## 检验结果

| 序列 | n | 均值 | 样本标准差 | ADF统计量 | ADF p值 | KPSS统计量 | KPSS p值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 澳大利亚死亡率 | {a['n']} | {a['mean']:.4f} | {a['sd']:.4f} | {a['adf_stat']:.4f} | {a['adf_p']:.4f} | {a['kpss_stat']:.4f} | {p_text(a)} |
| 密歇根湖水位 | {b['n']} | {b['mean']:.4f} | {b['sd']:.4f} | {b['adf_stat']:.4f} | {b['adf_p']:.4f} | {b['kpss_stat']:.4f} | {p_text(b)} |
| 水位一阶差分 | {diff['n']} | {diff['mean']:.4f} | {diff['sd']:.4f} | {diff['adf_stat']:.4f} | {diff['adf_p']:.2e} | {diff['kpss_stat']:.4f} | {p_text(diff)} |

### 1 澳大利亚死亡率

{a['stationarity']}

![澳大利亚死亡率时序图](figures/e2_7_time.png)

{a['correlation']}

![澳大利亚死亡率 ACF 与 PACF](figures/e2_7_acf_pacf.png)

PACF 第1、2、3阶分别为 {a['correlations'][1]['pacf']:.4f}、{a['correlations'][2]['pacf']:.4f}、{a['correlations'][3]['pacf']:.4f}；95%参考半宽约 {a['pacf_reference_halfwidth']:.4f}。第12阶为 {a['correlations'][12]['pacf']:.4f}。ACF 第1、5、10阶为 {a['correlations'][1]['acf']:.4f}、{a['correlations'][5]['acf']:.4f}、{a['correlations'][10]['acf']:.4f}，ACF进入区间不代表理论上变为零。

### 2 密歇根湖水位

{b['stationarity']}

![密歇根湖水位时序图](figures/e2_8_time.png)

{b['correlation']}

![密歇根湖水位 ACF 与 PACF](figures/e2_8_acf_pacf.png)

ACF 第1、5、10、16阶分别为 {b['correlations'][1]['acf']:.4f}、{b['correlations'][5]['acf']:.4f}、{b['correlations'][10]['acf']:.4f}、{b['correlations'][16]['acf']:.4f}。PACF 第1、2、3阶分别为 {b['correlations'][1]['pacf']:.4f}、{b['correlations'][2]['pacf']:.4f}、{b['correlations'][3]['pacf']:.4f}；95%参考半宽约 {b['pacf_reference_halfwidth']:.4f}。

### 补充 一阶差分验证

{diff['stationarity']}

![水位一阶差分时序图](figures/e2_8_diff_time.png)

![水位一阶差分 ACF 与 PACF](figures/e2_8_diff_acf_pacf.png)

{diff['correlation']}

## 文件说明

- `analysis.py`：生成全部分析图表、CSV、JSON 与本说明。
- `requirements.txt`：本实验依赖；`results/results.json` 记录实际运行版本和检验设置。
- `figures/`：6张分析图；`results/diagnostics.csv`：描述统计与平稳性检验；`results/*_correlations.csv`：各阶 ACF/PACF、区间与越界标记（0阶不参与特征判读）。
- `build_report.py`：根据计算结果生成本地 Word 报告，运行 `python exp3/build_report.py`；输出到 `report/`。
- 题目及 Word/PDF 报告仅保留本地，遵守项目隐私要求。数据表上传前仅清除作者等个人元数据，观测值保持不变。

## 方法参考

statsmodels 官方文档：[ACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html)、[PACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html)、[ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)、[KPSS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html)。
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main():
    FIGURES.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    results = {}
    for name, column, title, ylabel, start, end in DATASETS:
        frame = read_series(name, column, start, end)
        years, values = frame["year"].to_numpy(), frame[column].to_numpy()
        results[name.lower()] = analyze(years, values, title, ylabel, name.lower())
        if name == "E2_8":
            results["e2_8_diff"] = analyze(years[1:], np.diff(values), title + "一阶差分", "年度水位差", "e2_8_diff")
    results["settings"] = {"lags": LAGS, "alpha": ALPHA, "acf_ci": "Bartlett", "pacf_method": "ywm", "adf_regression": "c", "adf_autolag": "AIC", "kpss_regression": "c", "kpss_nlags": "auto", "versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "statsmodels": statsmodels.__version__}}
    (OUTPUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if isinstance(v, (str, int, float))} | {"series": name} for name, result in results.items() if name != "settings"]).to_csv(OUTPUT / "diagnostics.csv", index=False, float_format="%.10g")
    write_readme(results)
    for name in ("e2_7", "e2_8", "e2_8_diff"):
        r = results[name]
        print(f"{r['title']}: n={r['n']}, ADF p={r['adf_p']:.6g}, KPSS p{p_text(r)}")
    print("分析结果已保存到 exp3/figures/ 和 exp3/results/；说明见 exp3/README.md")


if __name__ == "__main__":
    main()
