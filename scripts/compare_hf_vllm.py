"""Compare local HF Transformers and vLLM real GPU benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


WORKLOAD_KEYS = (
    "prompt_type",
    "prompt_mode",
    "requests_per_level",
    "warmup",
    "runs",
    "max_tokens",
)
METRICS = ("tokens_per_second", "p95_ttft", "p95_tpot", "p95_latency")


def read_artifact(directory: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    with (directory / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (directory / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return metadata, rows


def aggregate(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["concurrency"])].append(row)
    return {
        concurrency: {
            metric: sum(float(row[metric]) for row in group) / len(group)
            for metric in METRICS
        }
        for concurrency, group in grouped.items()
    }


def change(before: float, after: float) -> float:
    return (after / before - 1) * 100 if before else 0.0


def memory_after(metadata: dict[str, Any]) -> str:
    environment = metadata.get("environment_after") or metadata.get("environment") or {}
    nvidia_smi = environment.get("nvidia_smi", {})
    if not isinstance(nvidia_smi, dict):
        return "unavailable"
    gpus = nvidia_smi.get("gpus", [])
    if isinstance(gpus, list) and gpus:
        gpu = gpus[0]
        return f"{gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB"
    stdout = nvidia_smi.get("stdout")
    return str(stdout) if stdout else "unavailable"


def write_report(
    path: Path,
    hf_meta: dict[str, Any],
    vllm_meta: dict[str, Any],
    hf: dict[int, dict[str, float]],
    vllm: dict[int, dict[str, float]],
) -> None:
    lines = [
        "# HF Transformers vs vLLM Real GPU Comparison",
        "",
        "> Positive delta means the vLLM value is higher than HF. For TTFT, TPOT and E2E latency, a negative delta is an improvement.",
        "",
        "## Controlled Variables",
        "",
    ]
    lines.extend(f"- `{key}`: `{hf_meta[key]}`" for key in WORKLOAD_KEYS)
    lines.extend(
        [
            f"- HF model: `{hf_meta.get('model')}`",
            f"- vLLM model: `{vllm_meta.get('model')}`",
            f"- HF backend: `{hf_meta.get('backend')}`",
            f"- vLLM backend: `{vllm_meta.get('backend')}`",
            f"- HF memory after run: `{memory_after(hf_meta)}`",
            f"- vLLM memory after run: `{memory_after(vllm_meta)}`",
            "",
            "## Results",
            "",
            "| Concurrency | Tokens/s HF -> vLLM | Delta | P95 TTFT HF -> vLLM | Delta | P95 TPOT HF -> vLLM | Delta | P95 E2E HF -> vLLM | Delta |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for concurrency in sorted(hf):
        base = hf[concurrency]
        served = vllm[concurrency]
        lines.append(
            "| {c} | {h_tokens:.2f} -> {v_tokens:.2f} | {d_tokens:+.1f}% | "
            "{h_ttft:.1f} -> {v_ttft:.1f} ms | {d_ttft:+.1f}% | "
            "{h_tpot:.2f} -> {v_tpot:.2f} ms | {d_tpot:+.1f}% | "
            "{h_e2e:.1f} -> {v_e2e:.1f} ms | {d_e2e:+.1f}% |".format(
                c=concurrency,
                h_tokens=base["tokens_per_second"],
                v_tokens=served["tokens_per_second"],
                d_tokens=change(base["tokens_per_second"], served["tokens_per_second"]),
                h_ttft=base["p95_ttft"] * 1000,
                v_ttft=served["p95_ttft"] * 1000,
                d_ttft=change(base["p95_ttft"], served["p95_ttft"]),
                h_tpot=base["p95_tpot"] * 1000,
                v_tpot=served["p95_tpot"] * 1000,
                d_tpot=change(base["p95_tpot"], served["p95_tpot"]),
                h_e2e=base["p95_latency"] * 1000,
                v_e2e=served["p95_latency"] * 1000,
                d_e2e=change(base["p95_latency"], served["p95_latency"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "HF Transformers is a local framework baseline: each concurrency level is implemented as local prompt batch size. vLLM is a serving system: each concurrency level is independent client requests through an OpenAI-compatible streaming endpoint. The comparison is useful for explaining why serving systems need scheduler, continuous batching, KV-cache block management and streaming observability, but it is not an identical transport-layer A/B.",
            "",
            "Expected interview explanation: HF is flexible and close to model code, while vLLM adds serving-time scheduling, token-level batching and PagedAttention-style KV Cache management. vLLM may improve throughput under concurrent workloads, but TTFT includes API, queueing and streaming overhead that local HF does not pay.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(
    path: Path,
    hf: dict[int, dict[str, float]],
    vllm: dict[int, dict[str, float]],
) -> None:
    concurrencies = sorted(hf)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    chart_metrics = (
        ("tokens_per_second", "Tokens/s", 1),
        ("p95_ttft", "P95 TTFT (ms)", 1000),
        ("p95_tpot", "P95 TPOT (ms)", 1000),
        ("p95_latency", "P95 E2E (ms)", 1000),
    )
    for axis, (metric, label, scale) in zip(axes.flat, chart_metrics):
        x = list(range(len(concurrencies)))
        width = 0.36
        axis.bar(
            [item - width / 2 for item in x],
            [hf[c][metric] * scale for c in concurrencies],
            width,
            label="HF Transformers",
        )
        axis.bar(
            [item + width / 2 for item in x],
            [vllm[c][metric] * scale for c in concurrencies],
            width,
            label="vLLM serving",
        )
        axis.set_xticks(x, [str(item) for item in concurrencies])
        axis.set_xlabel("Concurrency")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    figure.suptitle("HF Transformers vs vLLM")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HF Transformers and vLLM benchmark directories."
    )
    parser.add_argument("--hf", default="reports/hf_benchmark")
    parser.add_argument("--vllm", default="reports/gpu_benchmark")
    parser.add_argument("--output-dir", default="reports/hf_vs_vllm_comparison")
    args = parser.parse_args()

    hf_meta, hf_rows = read_artifact(Path(args.hf))
    vllm_meta, vllm_rows = read_artifact(Path(args.vllm))
    mismatches = [key for key in WORKLOAD_KEYS if hf_meta.get(key) != vllm_meta.get(key)]
    if mismatches:
        parser.error(f"workload mismatch: {', '.join(mismatches)}")
    hf = aggregate(hf_rows)
    vllm = aggregate(vllm_rows)
    if set(hf) != set(vllm):
        parser.error("concurrency levels differ")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_report(output_dir / "REPORT.md", hf_meta, vllm_meta, hf, vllm)
    write_plot(output_dir / "hf_vs_vllm.png", hf, vllm)
    print(f"saved_report={output_dir / 'REPORT.md'}")
    print(f"saved_plot={output_dir / 'hf_vs_vllm.png'}")


if __name__ == "__main__":
    main()
