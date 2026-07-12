"""Run a local Hugging Face Transformers baseline for LLM inference.

This benchmark is intentionally a local framework baseline. It does not expose
an HTTP service, does not implement continuous batching, and should not be read
as a production serving result. Its value is the controlled comparison against
the vLLM streaming benchmark under the same model family and workload.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import PromptMode, build_prompt, percentile
from benchmarks.gpu_report import write_plots


SUMMARY_FIELDS = [
    "run",
    "concurrency",
    "requests",
    "warmup",
    "prompt_type",
    "prompt_mode",
    "max_tokens",
    "success",
    "failed",
    "error_rate",
    "wall_time",
    "throughput",
    "tokens_per_second",
    "avg_prompt_tokens",
    "avg_completion_tokens",
    "p50_latency",
    "p95_latency",
    "p50_ttft",
    "p95_ttft",
    "p50_tpot",
    "p95_tpot",
    "first_error",
]


def parse_concurrencies(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("concurrency must contain positive integers")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def command_snapshot(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def gpu_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    snapshot = command_snapshot(command)
    gpus: list[dict[str, object]] = []
    if snapshot.get("returncode") == 0 and isinstance(snapshot.get("stdout"), str):
        for line in snapshot["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 7:
                continue
            name, driver, total, used, free, gpu_util, mem_util = parts
            gpus.append(
                {
                    "name": name,
                    "driver_version": driver,
                    "memory_total_mib": int(total),
                    "memory_used_mib": int(used),
                    "memory_free_mib": int(free),
                    "gpu_utilization_percent": int(gpu_util),
                    "memory_utilization_percent": int(mem_util),
                }
            )
    snapshot["gpus"] = gpus
    return snapshot


def environment_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
        },
        "nvidia_smi": gpu_snapshot(),
    }
    try:
        import torch

        snapshot["torch_cuda"] = {
            "available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except (ImportError, RuntimeError) as exc:
        snapshot["torch_cuda"] = {"error": str(exc)}
    return snapshot


def resolve_dtype(torch_module: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if value not in mapping:
        raise ValueError(f"unsupported dtype: {value}")
    return mapping[value]


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def gpu_line(environment: dict[str, object]) -> str:
    nvidia_smi = environment.get("nvidia_smi", {})
    if not isinstance(nvidia_smi, dict):
        return "unavailable"
    gpus = nvidia_smi.get("gpus", [])
    if isinstance(gpus, list) and gpus:
        gpu = gpus[0]
        return (
            f"{gpu['name']}, driver {gpu['driver_version']}, "
            f"{gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB used"
        )
    return str(nvidia_smi.get("stdout", "unavailable"))


def batch_prompts(
    prompt_type: str,
    prompt_mode: PromptMode,
    requests_count: int,
) -> list[str]:
    return [
        build_prompt(prompt_type, request_id, prompt_mode)
        for request_id in range(requests_count)
    ]


def chunks(items: list[str], size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def synchronize(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    prompts: list[str],
    request_offset: int,
    max_tokens: int,
    device: str,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_tokens = attention_mask.sum(dim=1).tolist()

    start = time.perf_counter()
    records: list[dict[str, Any]] = [
        {
            "request_id": request_offset + index,
            "ok": False,
            "latency": None,
            "ttft": None,
            "tpot": None,
            "prompt_tokens": int(prompt_tokens[index]),
            "completion_tokens": 0,
            "error": "",
        }
        for index in range(len(prompts))
    ]
    token_times: list[list[float]] = [[] for _ in prompts]
    finished = torch_module.zeros(len(prompts), dtype=torch_module.bool, device=device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    try:
        with torch_module.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            synchronize(torch_module)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            past_key_values = outputs.past_key_values
            first_token_at = time.perf_counter()

            for index, token_id in enumerate(next_token.tolist()):
                if eos_token_id is None or token_id != eos_token_id:
                    records[index]["ok"] = True
                    records[index]["ttft"] = first_token_at - start
                    records[index]["completion_tokens"] = 1
                    token_times[index].append(first_token_at)
                else:
                    finished[index] = True

            for _ in range(1, max_tokens):
                if bool(finished.all()):
                    break
                safe_next = next_token.masked_fill(finished, pad_token_id)
                attention_mask = torch_module.cat(
                    [
                        attention_mask,
                        torch_module.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                outputs = model(
                    input_ids=safe_next.unsqueeze(1),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                synchronize(torch_module)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                past_key_values = outputs.past_key_values
                token_at = time.perf_counter()

                for index, token_id in enumerate(next_token.tolist()):
                    if bool(finished[index]):
                        continue
                    if eos_token_id is not None and token_id == eos_token_id:
                        finished[index] = True
                        continue
                    records[index]["ok"] = True
                    records[index]["completion_tokens"] += 1
                    token_times[index].append(token_at)

        finished_at = time.perf_counter()
        for index, times in enumerate(token_times):
            if times:
                records[index]["latency"] = times[-1] - start
                records[index]["tpot"] = (
                    (times[-1] - times[0]) / (len(times) - 1)
                    if len(times) > 1
                    else None
                )
            else:
                records[index]["latency"] = finished_at - start
                records[index]["error"] = "generation produced no non-EOS token"
        return records
    except Exception as exc:
        failed_at = time.perf_counter()
        for record in records:
            record["latency"] = failed_at - start
            record["error"] = str(exc)
        return records


def mean_token_count(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_case(
    records: list[dict[str, Any]],
    *,
    concurrency: int,
    requests_count: int,
    warmup: int,
    prompt_type: str,
    prompt_mode: PromptMode,
    max_tokens: int,
    wall_time: float,
) -> dict[str, Any]:
    succeeded = [record for record in records if record["ok"]]
    failed = [record for record in records if not record["ok"]]
    latencies = [record["latency"] for record in succeeded if record["latency"] is not None]
    ttfts = [record["ttft"] for record in succeeded if record["ttft"] is not None]
    tpots = [record["tpot"] for record in succeeded if record["tpot"] is not None]
    prompt_tokens = [record["prompt_tokens"] for record in succeeded]
    completion_tokens = [record["completion_tokens"] for record in succeeded]
    return {
        "concurrency": concurrency,
        "requests": requests_count,
        "warmup": warmup,
        "prompt_type": prompt_type,
        "prompt_mode": prompt_mode.value,
        "max_tokens": max_tokens,
        "success": len(succeeded),
        "failed": len(failed),
        "error_rate": len(failed) / requests_count if requests_count else 0.0,
        "wall_time": wall_time,
        "throughput": len(succeeded) / wall_time if wall_time else 0.0,
        "tokens_per_second": sum(completion_tokens) / wall_time if wall_time else 0.0,
        "avg_prompt_tokens": mean_token_count(prompt_tokens),
        "avg_completion_tokens": mean_token_count(completion_tokens),
        "p50_latency": percentile(latencies, 0.50),
        "p95_latency": percentile(latencies, 0.95),
        "p50_ttft": percentile(ttfts, 0.50),
        "p95_ttft": percentile(ttfts, 0.95),
        "p50_tpot": percentile(tpots, 0.50),
        "p95_tpot": percentile(tpots, 0.95),
        "first_error": failed[0]["error"] if failed else "",
    }


def run_hf_case(
    *,
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    prompt_type: str,
    prompt_mode: PromptMode,
    concurrency: int,
    requests_count: int,
    warmup: int,
    max_tokens: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if warmup:
        warmup_prompts = batch_prompts(prompt_type, prompt_mode, warmup)
        for offset, prompt_batch in chunks(warmup_prompts, concurrency):
            generate_batch(
                model=model,
                tokenizer=tokenizer,
                torch_module=torch_module,
                prompts=prompt_batch,
                request_offset=-(offset + 1),
                max_tokens=max_tokens,
                device=device,
            )

    prompts = batch_prompts(prompt_type, prompt_mode, requests_count)
    records: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for offset, prompt_batch in chunks(prompts, concurrency):
        records.extend(
            generate_batch(
                model=model,
                tokenizer=tokenizer,
                torch_module=torch_module,
                prompts=prompt_batch,
                request_offset=offset,
                max_tokens=max_tokens,
                device=device,
            )
        )
    wall_time = time.perf_counter() - wall_start
    summary = summarize_case(
        records,
        concurrency=concurrency,
        requests_count=requests_count,
        warmup=warmup,
        prompt_type=prompt_type,
        prompt_mode=prompt_mode,
        max_tokens=max_tokens,
        wall_time=wall_time,
    )
    return summary, sorted(records, key=lambda record: record["request_id"])


def load_causal_lm(
    auto_model: Any,
    model_path: str,
    *,
    dtype: Any,
    trust_remote_code: bool,
) -> Any:
    kwargs = {"trust_remote_code": trust_remote_code}
    try:
        return auto_model.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return auto_model.from_pretrained(model_path, torch_dtype=dtype, **kwargs)


def write_report(path: Path, metadata_dict: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    environment = metadata_dict.get("environment_after") or metadata_dict["environment"]
    grouped = {
        concurrency: [row for row in rows if row["concurrency"] == concurrency]
        for concurrency in sorted({row["concurrency"] for row in rows})
    }
    lines = [
        "# Hugging Face Transformers Baseline Report",
        "",
        "> This is a local `transformers` baseline. It is not an HTTP serving benchmark and does not implement vLLM-style continuous batching.",
        "",
        "## Workload",
        "",
        f"- Model: `{metadata_dict['model']}`",
        f"- Prompt: `{metadata_dict['prompt_type']}` / `{metadata_dict['prompt_mode']}`",
        f"- Requests per concurrency: `{metadata_dict['requests_per_level']}`",
        f"- Warmup per run: `{metadata_dict['warmup']}`",
        f"- Repetitions: `{metadata_dict['runs']}`",
        f"- Max completion tokens: `{metadata_dict['max_tokens']}`",
        f"- Dtype: `{metadata_dict['dtype']}`",
        "",
        "## Environment",
        "",
        f"- Transformers: `{environment['packages']['transformers']}`",
        f"- PyTorch: `{environment['packages']['torch']}`",
        f"- GPU: `{gpu_line(environment)}`",
        f"- CUDA: `{environment['torch_cuda'].get('cuda_version')}`",
        "",
        "## Aggregate Results",
        "",
        "Each value is the arithmetic mean of the corresponding per-run result; P95 is computed within each run before averaging.",
        "",
        "| Concurrency | Success | Tokens/s | P95 TTFT (s) | P95 TPOT (s/token) | P95 E2E (s) | Error rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency, group in grouped.items():
        lines.append(
            "| {concurrency} | {success:.1f} | {tokens:.2f} | {ttft:.3f} | {tpot:.4f} | {latency:.3f} | {error:.2%} |".format(
                concurrency=concurrency,
                success=mean(group, "success"),
                tokens=mean(group, "tokens_per_second"),
                ttft=mean(group, "p95_ttft"),
                tpot=mean(group, "p95_tpot"),
                latency=mean(group, "p95_latency"),
                error=mean(group, "error_rate"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- HF baseline measures local greedy decode in one Python process.",
            "- `concurrency` means prompt batch size for each local generate wave, not independent HTTP clients.",
            "- TTFT is measured at the first decoded token after local prefill/decode synchronization; vLLM TTFT includes API, queueing, scheduling and streaming transport.",
            "- Use this report to explain why serving systems need batching, scheduling and KV-cache management; do not mix it with vLLM results as if both were identical service endpoints.",
            "",
            "Raw per-run summaries are in `summary.csv`; request-level samples are in `requests.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a local Hugging Face Transformers baseline."
    )
    parser.add_argument("--model", default=os.getenv("MODEL_PATH"))
    parser.add_argument("--prompt-type", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--prompt-mode", choices=[mode.value for mode in PromptMode], default="unique")
    parser.add_argument("--concurrency", type=parse_concurrencies, default=[1, 2, 4])
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", default="reports/hf_benchmark")
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required (or set MODEL_PATH)")
    if args.requests < 1 or args.warmup < 0 or args.max_tokens < 1 or args.runs < 1:
        parser.error("requests and max-tokens must be positive; warmup cannot be negative")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        parser.error(f"missing dependency: {exc}")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but torch.cuda.is_available() is false")

    environment_before = environment_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = resolve_dtype(torch, args.dtype)
    model = load_causal_lm(
        AutoModelForCausalLM,
        args.model,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(args.device)
    model.eval()

    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    mode = PromptMode(args.prompt_mode)
    for run in range(1, args.runs + 1):
        for concurrency in args.concurrency:
            summary, case_records = run_hf_case(
                model=model,
                tokenizer=tokenizer,
                torch_module=torch,
                prompt_type=args.prompt_type,
                prompt_mode=mode,
                concurrency=concurrency,
                requests_count=args.requests,
                warmup=args.warmup,
                max_tokens=args.max_tokens,
                device=args.device,
            )
            summaries.append({"run": run, **summary})
            records.extend(
                {"run": run, "concurrency": concurrency, **record}
                for record in case_records
            )
            print(
                "run={run} concurrency={concurrency} success={success}/{requests} "
                "throughput={throughput:.2f} req/s tokens/s={tokens_per_second:.2f} "
                "p95_ttft={p95_ttft:.3f}s p95_tpot={p95_tpot:.4f}s "
                "p95_e2e={p95_latency:.3f}s".format(run=run, **summary),
                flush=True,
            )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    raw_fields = [
        "run",
        "concurrency",
        "request_id",
        "ok",
        "latency",
        "ttft",
        "tpot",
        "prompt_tokens",
        "completion_tokens",
        "error",
    ]
    write_csv(output_dir / "requests.csv", records, raw_fields)
    metadata_dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "local Hugging Face Transformers generate baseline",
        "experiment_variant": "hf_transformers_baseline",
        "model": args.model,
        "prompt_type": args.prompt_type,
        "prompt_mode": args.prompt_mode,
        "concurrency": args.concurrency,
        "requests_per_level": args.requests,
        "warmup": args.warmup,
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "device": args.device,
        "notes": "Concurrency means local batch size; TTFT is local first-token timing, not streaming transport TTFT.",
        "environment": environment_before,
        "environment_before": environment_before,
        "environment_after": environment_snapshot(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_dict, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "REPORT.md", metadata_dict, summaries)
    plot_paths = write_plots(summaries, records, output_dir)
    print("saved_plots=" + ", ".join(str(path) for path in plot_paths))
    print(f"saved_results={output_dir}")


if __name__ == "__main__":
    main()
