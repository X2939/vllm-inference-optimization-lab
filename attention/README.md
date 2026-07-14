# Attention Educational Examples

This directory contains small, legacy educational examples for understanding
attention mechanics. They are not the project evidence for GPU performance and
are not used by the default inference engine benchmark.

- `naive_attention.py`: pure-Python textbook scaled dot-product attention. It
  materializes the full attention score matrix and is useful for explaining why
  standard attention has quadratic intermediate memory.
- `flash_attention.py`: pure-Python tiling and online-softmax sketch. It shows
  the idea behind FlashAttention, but it is not a CUDA kernel and is not a
  substitute for a real GPU backend benchmark.

For the project's interview-facing kernel experiment, use:

```bash
make attention-kernel-probe
```

That command runs `scripts/attention_kernel_probe.py`, which compares PyTorch
CUDA SDPA `math` and `flash` backends on the same Q/K/V shapes and writes the
report to `reports/attention_kernel_probe/`.
