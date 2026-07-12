# Backend Boundary Study

This note explains why the repository contains PyTorch, ONNX Runtime, TensorRT
and vLLM related paths without treating all of them as equivalent production LLM
benchmark evidence.

## Layers

| Path | Main role | What this project uses it for | Evidence level |
|---|---|---|---|
| Hugging Face Transformers / PyTorch | Local model execution and development baseline | Same Qwen-family model and workload, local greedy decode baseline | Real local GPU baseline |
| vLLM | Online serving, scheduler, continuous batching, KV Cache and streaming API | Real OpenAI-compatible benchmark with TTFT, TPOT, P95, throughput and memory | Main real GPU serving evidence |
| ONNX Runtime | Graph/runtime execution backend | Educational adapter around a simplified exported model | Backend boundary example |
| TensorRT | NVIDIA optimized engine runtime | Educational adapter around a simplified engine | Backend boundary example |

## Why HF vs vLLM Is the Main Upgrade

The most useful comparison for this interview project is local HF Transformers
against vLLM serving:

- HF is close to model code and good for correctness checks and a baseline.
- vLLM adds serving-time scheduling, continuous batching and PagedAttention-style
  KV-cache block management.
- The same prompt workload can show why online inference is not just a faster
  `forward()` call. Queueing, batching, prefill/decode separation and streaming
  observability matter.

The comparison still has a boundary: HF `concurrency` is local batch size, while
vLLM `concurrency` is independent streaming client requests. Use the comparison
to explain system design tradeoffs, not as a perfect transport-layer A/B.

## Why ONNX/TensorRT Stay Educational

The existing ONNX and TensorRT executors use simplified model artifacts under
`models/`. They are useful for understanding adapter boundaries:

```text
Scheduler -> Worker -> ModelRunner -> backend executor
```

They should not be described as full Qwen LLM backend performance results. A
real TensorRT-LLM comparison would require a separate engine build flow,
precision policy, dynamic shape profiles, plugin/kernel compatibility checks and
KV-cache handling. That is a different project scope.

## Interview Positioning

Use this wording:

> My main measured serving path is vLLM + Qwen. I also added a local HF
> Transformers baseline to avoid treating vLLM as a black box. ONNX Runtime and
> TensorRT are kept as backend-boundary examples, but I do not mix their
> simplified-model results with real Qwen/vLLM numbers.

This framing shows breadth without overstating the evidence.
