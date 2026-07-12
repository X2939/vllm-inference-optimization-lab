# GPU Operator Notes

The main project is still an inference-serving project. This note adds a small
bottom-up layer so the serving metrics can be connected to GPU execution.

## Why This Matters

LLM inference performance is shaped by two layers at the same time:

- Serving layer: scheduler, continuous batching, KV Cache, PagedAttention block
  management, prefix cache and request queueing.
- Kernel layer: attention implementation, memory traffic, dtype, tensor shape,
  CUDA/PyTorch backend selection and GPU architecture.

TTFT is often dominated by queueing plus prefill. TPOT is closer to the repeated
decode path, where attention and KV-cache reads matter heavily. That is why a
serving project should be able to explain both layers, even if it does not write
custom CUDA kernels.

## What the Probe Does

`scripts/attention_kernel_probe.py` runs PyTorch CUDA
`scaled_dot_product_attention` with explicit backend selection:

- `math`: conservative reference path.
- `flash`: fused FlashAttention-style SDPA backend when available.
- `mem_efficient`: memory-efficient fused backend when available.

It sweeps sequence length and records:

- P50/P95 latency.
- Peak allocated GPU memory.
- Numerical difference against the math backend.
- Backend availability errors.

Run it with:

```bash
make attention-kernel-probe
```

Outputs are written to `reports/attention_kernel_probe/`.

## Interview Boundary

This is not a claim that the project implements a custom FlashAttention CUDA
kernel. The correct statement is:

> I added a PyTorch CUDA SDPA backend probe to connect serving metrics with
> lower-level attention execution. It compares math, flash and memory-efficient
> SDPA paths on the same Q/K/V shapes, records latency, peak memory and numerical
> difference, and helps explain why attention kernel choice affects TPOT and
> throughput.

This is enough to show bottom-layer awareness without overstating kernel
engineering experience.
