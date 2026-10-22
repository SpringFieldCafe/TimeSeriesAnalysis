"""从本地教学 Excel 提取纯数据 CSV，保留原文件；日常分析不依赖 Excel。"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCES = [("E6_2", "1973-01-01", "1978-12-01", "MS", 72),
           ("E6_7", "1948-01-01", "1979-10-01", "QS-JAN", 128)]


def main():
    audit = []
    for name, start, end, frequency, count in SOURCES:
        source = ROOT / "doc" / f"{name}.xlsx"
        book = pd.ExcelFile(source)
        frame = pd.read_excel(book, sheet_name="Sheet1")
        assert list(frame.columns) == ["time", "x"]
        assert len(frame) == count and not frame.isna().any().any()
        np.testing.assert_array_equal(frame.time, pd.date_range(start, end, freq=frequency))
        assert (frame.x > 0).all() and (frame.x == frame.x.astype(int)).all()
        for sheet in book.sheet_names[1:]:
            assert pd.read_excel(book, sheet_name=sheet).empty
        output = source.with_suffix(".csv")
        frame.to_csv(output, index=False, date_format="%Y-%m-%d", lineterminator="\n")
        reread = pd.read_csv(output, parse_dates=["time"])
        pd.testing.assert_frame_equal(frame, reread)
        audit.append({"source": source.name, "sheet": "Sheet1", "csv": output.name,
                      "rows": count, "start": start, "end": end,
                      "values_verified": True,
                      "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "source_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("已逐值核对两份 CSV：72 个月、128 个季度；原始 Excel 保留不变。")


if __name__ == "__main__":
    main()
