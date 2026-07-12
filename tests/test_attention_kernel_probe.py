from __future__ import annotations

import argparse
import csv

from scripts.attention_kernel_probe import (
    SUMMARY_FIELDS,
    parse_csv_ints,
    percentile,
    write_csv,
    write_report,
)


def test_parse_csv_ints_rejects_invalid_values() -> None:
    assert parse_csv_ints("128,256") == [128, 256]
    try:
        parse_csv_ints("0")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("expected ArgumentTypeError")


def test_percentile_interpolates_probe_values() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 3.8499999999999996


def test_probe_writers_create_csv_and_report(tmp_path) -> None:
    rows = [
        {
            "backend": "math",
            "seq_len": 128,
            "batch_size": 1,
            "num_heads": 8,
            "head_dim": 64,
            "dtype": "fp16",
            "causal": True,
            "success": True,
            "latency_ms": 1.0,
            "p50_ms": 1.0,
            "p95_ms": 1.2,
            "peak_memory_mib": 16.0,
            "max_abs_diff_vs_math": 0.0,
            "error": "",
        }
    ]
    csv_path = tmp_path / "summary.csv"
    report_path = tmp_path / "REPORT.md"
    metadata = {
        "seq_lens": [128],
        "batch_size": 1,
        "num_heads": 8,
        "head_dim": 64,
        "dtype": "fp16",
        "causal": True,
        "warmup": 1,
        "runs": 1,
    }

    write_csv(csv_path, rows)
    write_report(report_path, rows, metadata)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert list(csv_rows[0]) == SUMMARY_FIELDS
    assert "Attention Kernel Probe" in report_path.read_text(encoding="utf-8")
