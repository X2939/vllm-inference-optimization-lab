# Prefill/Decode Disaggregation Study

This note is a design study for AI Infra interviews. It explains how
prefill/decode (PD) disaggregation relates to this repository's existing
Scheduler, KV Cache, PagedAttention-style block table, Prefix Cache, and
benchmark work. It is not implemented as a runnable multi-node serving path in
this project.

## Why Split Prefill and Decode

LLM inference has two phases with different resource profiles.

| Phase | What Happens | Typical Bottleneck | Serving Metric Most Affected |
|---|---|---|---|
| Prefill | Process the full prompt, build Q/K/V, compute prompt attention, write KV Cache, sample the first token | Large matrix compute and prompt length | TTFT |
| Decode | Generate one token per request per step while repeatedly reading historical KV | KV reads, memory bandwidth, batch shape, kernel efficiency | TPOT / ITL, throughput, P95 |

In a single vLLM instance, prefill and decode share one scheduler and one GPU
resource pool. Long prompts can occupy compute and delay interactive decode
steps; decode-heavy traffic can also make it harder to batch prefill efficiently.

PD disaggregation separates the phases into different resource pools:

```text
Client / Router
      |
      v
Prefill Worker Pool
  - runs prompt prefill
  - creates KV Cache
  - returns first token / metadata
      |
      | KV transfer
      v
Decode Worker Pool
  - receives KV Cache
  - runs token-by-token decode
  - streams output tokens
```

The goal is not "always faster." The goal is to let prefill and decode be
scaled, scheduled, and tuned independently when the workload makes their
interference expensive.

## Relation to This Project

This repository already models the two phases inside one process:

```text
Scheduler.schedule()
    -> prefill requests may receive many scheduled prompt tokens
    -> decode requests usually receive one scheduled token each
ModelRunner.execute_model()
Scheduler.update_from_output()
    -> first token moves PREFILL -> DECODE
    -> generated tokens append KV
```

The current project is an aggregated design:

```text
one Scheduler + one Worker + one KVCacheManager
```

A disaggregated design would split the ownership boundaries:

| Current Component | Aggregated Role | PD Disaggregation Counterpart |
|---|---|---|
| Scheduler | Schedules prefill and decode in one token budget | Router plus separate prefill/decode schedulers |
| KVCacheManager | Allocates, shares, appends, and frees local KV blocks | Local KV manager plus remote KV transfer protocol |
| PrefixCache | Reuses block-aligned prompt KV in one process | Needs prefix-aware routing or distributed KV lookup |
| ModelRunner | Simulates both prefill and decode execution | Prefill runner and decode runner can use different parallelism |
| Benchmark | Measures TTFT, TPOT, P95, throughput in one endpoint | Must also measure KV transfer time and queueing per pool |

## KV Transfer Is the Hard Part

The prefill worker produces KV blocks that the decode worker must consume. That
handoff adds a new cost:

```text
PD latency = prefill queue
           + prefill compute
           + KV transfer
           + decode queue
           + decode steps
```

PD separation helps only when the benefits of independent scheduling and scaling
outweigh KV transfer overhead.

Key questions:

- How large is the KV payload for the model, dtype, layer count, and sequence
  length?
- Is the transfer local GPU-to-GPU, same node, cross-node RDMA, or TCP?
- Can the decode worker start before the full transfer completes?
- Does the router know which decode worker already holds useful prefix KV?
- How do retries and failures avoid leaking or duplicating KV blocks?

The KV size intuition is the same as in this project:

```text
KV bytes ~= 2 * layers * kv_heads * head_dim * sequence_length * bytes_per_element
```

The factor `2` is for K and V. Long prompts and high concurrency quickly make KV
transfer a real systems problem, not just a function call.

## How It Interacts With Prefix Cache and PagedAttention

PagedAttention-style block management and Prefix Cache remain useful, but their
scope changes.

```text
PagedAttention:
  manages physical KV blocks and logical block tables.
  In PD, the block table or an equivalent KV descriptor must cross the worker
  boundary.

Prefix Cache:
  skips repeated prefill for shared prefixes.
  In PD, the router may need prefix-aware routing so requests with the same
  prefix land on workers that already have the relevant KV.
```

This distinction is important:

- PagedAttention solves KV allocation and fragmentation.
- Prefix Cache solves repeated prefix computation.
- PD separation solves phase interference and resource specialization.

They are complementary, not substitutes.

## When PD Separation Helps

PD disaggregation is most attractive when:

- Prompts are long and prefill creates TTFT spikes for interactive decode
  traffic.
- Traffic mix changes over time, so prefill and decode need different scaling
  ratios.
- Hardware is heterogeneous and prefill/decode can be assigned to different GPU
  types or parallelism strategies.
- Prefix reuse, multi-turn chat, or agent workflows make routing/cache placement
  important.

It may not help when:

- Requests are short and KV transfer overhead dominates.
- The service runs on one small GPU where separate pools cannot be provisioned.
- Network bandwidth is weak relative to KV payload size.
- The bottleneck is outside prefill/decode execution, such as tokenizer, HTTP,
  sampling, or application logic.

## Benchmark Plan for a Future Implementation

If this repository were extended beyond a design note, the benchmark should add
PD-specific fields instead of only comparing end-to-end latency.

| Metric | Why It Matters |
|---|---|
| Prefill queue time | Whether prefill workers are saturated |
| Prefill compute time | Prompt processing cost |
| KV transfer bytes/time | Whether disaggregation introduces a new bottleneck |
| Decode queue time | Whether decode workers are saturated |
| TTFT | User-visible first-token delay |
| TPOT / ITL | Decode smoothness after first token |
| P95 / P99 E2E | Tail behavior under mixed traffic |
| Cache hit / routing hit rate | Whether prefix-aware placement is effective |

Suggested controlled workloads:

1. Short unique prompts: PD should not be assumed to help.
2. Long unique prompts: observe prefill pressure and transfer overhead.
3. Shared-prefix prompts: combine Prefix Cache with prefix-aware routing.
4. Mixed traffic: long summarization prompts plus short chat prompts.

## Interview-Safe Explanation

> Prefill/decode disaggregation separates prompt processing and token-by-token
> generation into different worker pools. Prefill is more compute-heavy because
> the full prompt is known and processed in parallel; decode is more
> memory-bandwidth-sensitive because each step reads historical KV and generates
> one token per active request. Splitting them can reduce interference and let us
> scale/tune the two phases independently. The hard part is KV transfer: prefill
> produces KV Cache that decode must consume, so transfer latency, bandwidth,
> placement, and failure handling determine whether PD separation is actually a
> win. In my current project I do not implement multi-node PD serving; I treat it
> as the next system extension after Scheduler, KV Cache, Prefix Cache, and real
> GPU benchmark are already in place.

## References

- vLLM documentation, "Disaggregated Prefilling (experimental)": separate
  prefill and decode instances, connected through KV transfer under
  `vllm/distributed/kv_transfer`.
- BentoML LLM Inference Handbook, "Prefill-decode disaggregation": prefill and
  decode have different resource profiles and can be scaled independently.
- Ray Serve LLM guide, "Prefill/decode disaggregation": deployment-level
  serving pattern and KV transfer backend considerations.
