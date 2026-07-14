"""Probe PyTorch CUDA scaled-dot-product attention backends.

This is a small GPU-kernel-facing experiment. It does not implement a custom
CUDA kernel; instead it exercises PyTorch SDPA backend selection so the project
can connect serving metrics to lower-level attention execution choices.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt


SUMMARY_FIELDS = [
    "backend",
    "seq_len",
    "batch_size",
    "num_heads",
    "head_dim",
    "dtype",
    "causal",
    "success",
    "latency_ms",
    "p50_ms",
    "p95_ms",
    "peak_memory_mib",
    "max_abs_diff_vs_math",
    "error",
]


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("value must contain positive integers")
    return result


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def dtype_from_name(torch_module: Any, value: str) -> Any:
    mapping = {
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    if value not in mapping:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")
    return mapping[value]


@contextmanager
def sdpa_backend_context(torch_module: Any, backend: str) -> Iterator[None]:
    """Select one SDPA backend across PyTorch API variants."""

    if hasattr(torch_module.nn, "attention") and hasattr(
        torch_module.nn.attention, "sdpa_kernel"
    ):
        sdpa_kernel = torch_module.nn.attention.sdpa_kernel
        sdpa_backend = torch_module.nn.attention.SDPBackend
        backend_map = {
            "math": sdpa_backend.MATH,
            "flash": sdpa_backend.FLASH_ATTENTION,
        }
        with sdpa_kernel([backend_map[backend]]):
            yield
        return

    if hasattr(torch_module.backends.cuda, "sdp_kernel"):
        flags = {
            "enable_math": backend == "math",
            "enable_flash": backend == "flash",
            "enable_mem_efficient": False,
        }
        with torch_module.backends.cuda.sdp_kernel(**flags):
            yield
        return

    raise RuntimeError("PyTorch SDPA backend selection API is unavailable")


def attention_call(torch_module: Any, q: Any, k: Any, v: Any, causal: bool) -> Any:
    return torch_module.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )


def synchronize(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def make_qkv(
    torch_module: Any,
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: Any,
    device: str,
    seed: int,
) -> tuple[Any, Any, Any]:
    generator = torch_module.Generator(device=device)
    generator.manual_seed(seed)
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch_module.randn(shape, device=device, dtype=dtype, generator=generator)
    k = torch_module.randn(shape, device=device, dtype=dtype, generator=generator)
    v = torch_module.randn(shape, device=device, dtype=dtype, generator=generator)
    return q, k, v


def benchmark_backend(
    torch_module: Any,
    *,
    backend: str,
    q: Any,
    k: Any,
    v: Any,
    causal: bool,
    warmup: int,
    runs: int,
) -> tuple[dict[str, Any], Any | None]:
    try:
        gc.collect()
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
        output = None
        with sdpa_backend_context(torch_module, backend):
            for _ in range(warmup):
                output = attention_call(torch_module, q, k, v, causal)
            synchronize(torch_module)

            timings: list[float] = []
            for _ in range(runs):
                start = torch_module.cuda.Event(enable_timing=True)
                end = torch_module.cuda.Event(enable_timing=True)
                start.record()
                output = attention_call(torch_module, q, k, v, causal)
                end.record()
                synchronize(torch_module)
                timings.append(float(start.elapsed_time(end)))

        peak_memory_mib = torch_module.cuda.max_memory_allocated() / (1024 * 1024)
        return (
            {
                "success": True,
                "latency_ms": sum(timings) / len(timings),
                "p50_ms": percentile(timings, 0.50),
                "p95_ms": percentile(timings, 0.95),
                "peak_memory_mib": peak_memory_mib,
                "error": "",
            },
            output.detach() if output is not None else None,
        )
    except Exception as exc:
        return (
            {
                "success": False,
                "latency_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "peak_memory_mib": 0.0,
                "error": str(exc),
            },
            None,
        )


def run_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"missing dependency: {exc}") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")

    dtype = dtype_from_name(torch, args.dtype)
    rows: list[dict[str, Any]] = []
    for seq_len in args.seq_lens:
        q, k, v = make_qkv(
            torch,
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            seq_len=seq_len,
            head_dim=args.head_dim,
            dtype=dtype,
            device=args.device,
            seed=args.seed + seq_len,
        )
        math_output = None
        for backend in args.backends:
            metrics, output = benchmark_backend(
                torch,
                backend=backend,
                q=q,
                k=k,
                v=v,
                causal=args.causal,
                warmup=args.warmup,
                runs=args.runs,
            )
            if backend == "math" and output is not None:
                math_output = output.float()
            max_abs_diff = None
            if math_output is not None and output is not None:
                max_abs_diff = float((output.float() - math_output).abs().max().item())
            row = {
                "backend": backend,
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,
                "causal": args.causal,
                "max_abs_diff_vs_math": max_abs_diff,
                **metrics,
            }
            rows.append(row)
            status = "ok" if row["success"] else "failed"
            print(
                "backend={backend} seq={seq_len} status={status} "
                "latency={latency_ms:.3f}ms p95={p95_ms:.3f}ms "
                "peak={peak_memory_mib:.1f}MiB diff={diff}".format(
                    backend=backend,
                    seq_len=seq_len,
                    status=status,
                    latency_ms=float(row["latency_ms"]),
                    p95_ms=float(row["p95_ms"]),
                    peak_memory_mib=float(row["peak_memory_mib"]),
                    diff=row["max_abs_diff_vs_math"],
                ),
                flush=True,
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row["success"]]
    if not successful:
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        grouped[str(row["backend"])].append(row)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for backend, backend_rows in sorted(grouped.items()):
        ordered = sorted(backend_rows, key=lambda row: int(row["seq_len"]))
        x = [int(row["seq_len"]) for row in ordered]
        axes[0].plot(x, [float(row["p95_ms"]) for row in ordered], "o-", label=backend)
        axes[1].plot(
            x,
            [float(row["peak_memory_mib"]) for row in ordered],
            "o-",
            label=backend,
        )
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("P95 latency (ms)")
    axes[0].set_title("SDPA backend latency")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Peak allocated memory (MiB)")
    axes[1].set_title("SDPA backend memory")
    axes[1].grid(alpha=0.25)
    axes[0].legend(loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, Any]], metadata_dict: dict[str, Any]) -> None:
    lines = [
        "# Attention Kernel Probe",
        "",
        "> This probe uses PyTorch CUDA SDPA backend selection. It is a lower-level execution experiment, not a custom CUDA kernel implementation.",
        "",
        "## Workload",
        "",
        f"- Sequence lengths: `{metadata_dict['seq_lens']}`",
        f"- Batch size: `{metadata_dict['batch_size']}`",
        f"- Heads: `{metadata_dict['num_heads']}`",
        f"- Head dim: `{metadata_dict['head_dim']}`",
        f"- Dtype: `{metadata_dict['dtype']}`",
        f"- Causal: `{metadata_dict['causal']}`",
        f"- Warmup / runs: `{metadata_dict['warmup']}` / `{metadata_dict['runs']}`",
        "",
        "## Results",
        "",
        "| Backend | Seq | Success | P95 latency (ms) | Peak memory (MiB) | Max abs diff vs math | Error |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        diff = row["max_abs_diff_vs_math"]
        diff_text = "" if diff is None else f"{float(diff):.4g}"
        error = str(row["error"]).replace("|", "/")
        lines.append(
            "| {backend} | {seq} | {success} | {p95:.3f} | {mem:.1f} | {diff} | {error} |".format(
                backend=row["backend"],
                seq=row["seq_len"],
                success=row["success"],
                p95=float(row["p95_ms"]),
                mem=float(row["peak_memory_mib"]),
                diff=diff_text,
                error=error,
            )
        )
    lines.extend(
        [
            "",
            "## How to Read It",
            "",
            "- `math` is the conservative reference path and is useful for numerical comparison.",
            "- `flash` attempts to use the fused FlashAttention-style SDPA kernel when the GPU, dtype and shape support it.",
            "- Lower latency usually comes from avoiding materializing the full attention matrix and reducing HBM traffic, but backend availability depends on CUDA, PyTorch, GPU architecture, dtype, head dimension and mask pattern.",
            "",
            "This result complements the vLLM benchmark: serving metrics such as TPOT and throughput are affected by scheduler and KV Cache behavior, but the decode/prefill compute path still depends on attention kernel efficiency.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def environment_snapshot() -> dict[str, Any]:
    try:
        import torch

        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "torch": package_version("torch"),
                "triton": package_version("triton"),
            },
            "torch_cuda": {
                "available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
    except (ImportError, RuntimeError) as exc:
        return {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe PyTorch CUDA SDPA attention backends."
    )
    parser.add_argument("--seq-lens", type=parse_csv_ints, default=[128, 256, 512, 1024])
    parser.add_argument("--backends", nargs="+", default=["math", "flash"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="reports/attention_kernel_probe")
    args = parser.parse_args()

    allowed = {"math", "flash"}
    invalid = sorted(set(args.backends) - allowed)
    if invalid:
        parser.error(f"unsupported backends: {', '.join(invalid)}")
    if args.batch_size < 1 or args.num_heads < 1 or args.head_dim < 1:
        parser.error("batch-size, num-heads and head-dim must be positive")
    if args.warmup < 0 or args.runs < 1:
        parser.error("warmup cannot be negative and runs must be positive")

    output_dir = Path(args.output_dir)
    rows = run_probe(args)
    metadata_dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seq_lens": args.seq_lens,
        "backends": args.backends,
        "batch_size": args.batch_size,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "device": args.device,
        "warmup": args.warmup,
        "runs": args.runs,
        "causal": args.causal,
        "environment": environment_snapshot(),
    }
    write_csv(output_dir / "summary.csv", rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_dict, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_plot(output_dir / "attention_kernel_probe.png", rows)
    write_report(output_dir / "REPORT.md", rows, metadata_dict)
    print(f"saved_results={output_dir}")


if __name__ == "__main__":
    main()
