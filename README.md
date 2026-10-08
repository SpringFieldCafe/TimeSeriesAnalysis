# 时间序列分析实验

本仓库用于完成时间序列分析实验，实验一的脚本、原始数据和图表位于 `exp1/`。实验报告 Word、PDF 文档以及本机配置和个人敏感信息由 `.gitignore` 排除。

## 实验一

在已安装 pandas、matplotlib、statsmodels、openpyxl 的 Python 环境中运行：

```bash
python exp1/analysis.py
```

脚本读取 `exp1/doc/` 中的 Excel 数据，并将四张时序图和样本自相关图保存到 `exp1/figures/`。终端会给出销售量序列的 Ljung–Box 检验结果。

依赖：pandas、matplotlib、statsmodels、openpyxl。无需额外配置环境；若需安装依赖，可按本机 Anaconda 环境管理方式安装。
