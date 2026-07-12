from __future__ import annotations

import argparse
import csv

import pytest

from benchmarks.common import PromptMode
from scripts.compare_hf_vllm import aggregate, change, memory_after, write_report
from scripts.hf_benchmark import (
    batch_prompts,
    parse_concurrencies,
    summarize_case,
    write_csv,
)


def test_parse_concurrencies_rejects_empty_values() -> None:
    assert parse_concurrencies("1,2,4") == [1, 2, 4]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_concurrencies("")


def test_batch_prompts_can_build_shared_prefix_workload() -> None:
    prompts = batch_prompts("long", PromptMode.SHARED_PREFIX, 2)

    assert len(prompts) == 2
    assert all("共享前缀" in prompt for prompt in prompts)


def test_summarize_case_computes_streaming_metrics() -> None:
    records = [
        {
            "ok": True,
            "latency": 1.0,
            "ttft": 0.2,
            "tpot": 0.1,
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "error": "",
        },
        {
            "ok": False,
            "latency": 0.5,
            "ttft": None,
            "tpot": None,
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "error": "boom",
        },
    ]

    summary = summarize_case(
        records,
        concurrency=2,
        requests_count=2,
        warmup=1,
        prompt_type="short",
        prompt_mode=PromptMode.UNIQUE,
        max_tokens=8,
        wall_time=1.5,
    )

    assert summary["success"] == 1
    assert summary["failed"] == 1
    assert summary["tokens_per_second"] == pytest.approx(8 / 1.5)
    assert summary["first_error"] == "boom"


def test_write_csv_uses_expected_fields(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    write_csv(path, [{"a": 1, "b": 2, "ignored": 3}], ["a", "b"])

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"a": "1", "b": "2"}]


def test_compare_helpers_aggregate_and_write_report(tmp_path) -> None:
    rows = [
        {
            "concurrency": "1",
            "tokens_per_second": "10",
            "p95_ttft": "0.2",
            "p95_tpot": "0.1",
            "p95_latency": "1.0",
        },
        {
            "concurrency": "1",
            "tokens_per_second": "20",
            "p95_ttft": "0.4",
            "p95_tpot": "0.2",
            "p95_latency": "2.0",
        },
    ]
    aggregated = aggregate(rows)

    assert aggregated[1]["tokens_per_second"] == 15
    assert change(10, 15) == 50
    assert (
        memory_after(
            {
                "environment_after": {
                    "nvidia_smi": {
                        "gpus": [{"memory_used_mib": 1, "memory_total_mib": 2}]
                    }
                }
            }
        )
        == "1 / 2 MiB"
    )

    meta = {
        "prompt_type": "medium",
        "prompt_mode": "unique",
        "requests_per_level": 2,
        "warmup": 1,
        "runs": 1,
        "max_tokens": 8,
        "model": "test-model",
        "backend": "test-backend",
    }
    report_path = tmp_path / "REPORT.md"
    write_report(report_path, meta, meta, aggregated, aggregated)

    content = report_path.read_text(encoding="utf-8")
    assert "HF Transformers vs vLLM" in content
    assert "Interpretation Boundary" in content
