PYTHON ?= python3
VLLM_ENV ?= /home/xxx/venvs/vllm-env
VLLM_PYTHON ?= $(VLLM_ENV)/bin/python
VLLM_BIN ?= $(VLLM_ENV)/bin/vllm
VLLM_HOST ?= 0.0.0.0
VLLM_PORT ?= 8000
API_HOST ?= 0.0.0.0
API_PORT ?= 9000
MODEL_PATH ?= /home/xxx/models/Qwen2.5-1.5B-Instruct
MODEL_AWQ_PATH ?= /home/xxx/models/Qwen2.5-1.5B-Instruct-AWQ
MODEL_NAME ?= $(MODEL_PATH)
API_KEY ?= token-abc123
GPU_BENCH_CONCURRENCY ?= 1,2,4
GPU_BENCH_REQUESTS ?= 20
GPU_BENCH_WARMUP ?= 3
GPU_BENCH_RUNS ?= 3
GPU_BENCH_MAX_TOKENS ?= 128
GPU_BENCH_OUTPUT ?= reports/gpu_benchmark
GPU_BENCH_VARIANT ?= bf16_baseline
GPU_BENCH_SERVER_ARGS ?= effective: enable_prefix_caching=true, enable_chunked_prefill=true
SERVER_EXTRA_ARGS ?=
GPU_PREFIX_CONCURRENCY ?= 1,2,4
GPU_PREFIX_REQUESTS ?= 16
GPU_PREFIX_WARMUP ?= 3
GPU_PREFIX_RUNS ?= 3
GPU_PREFIX_MAX_TOKENS ?= 64
GPU_PREFIX_OUTPUT ?= reports/gpu_prefix_cache
GPU_PREFIX_VARIANT ?= unspecified
GPU_PREFIX_SERVER_ARGS ?=
GPU_AWQ_OUTPUT ?= reports/gpu_awq
QUALITY_OUTPUT ?= reports/quality_smoke
QUALITY_MAX_TOKENS ?= 96
HF_BENCH_OUTPUT ?= reports/hf_benchmark
HF_BENCH_CONCURRENCY ?= $(GPU_BENCH_CONCURRENCY)
HF_BENCH_REQUESTS ?= $(GPU_BENCH_REQUESTS)
HF_BENCH_WARMUP ?= $(GPU_BENCH_WARMUP)
HF_BENCH_RUNS ?= $(GPU_BENCH_RUNS)
HF_BENCH_MAX_TOKENS ?= $(GPU_BENCH_MAX_TOKENS)
HF_BENCH_DTYPE ?= auto
HF_BENCH_DEVICE ?= cuda
ATTN_PROBE_OUTPUT ?= reports/attention_kernel_probe
ATTN_PROBE_SEQ_LENS ?= 128,256,512,1024
ATTN_PROBE_BACKENDS ?= math flash
ATTN_PROBE_DTYPE ?= fp16
ATTN_PROBE_RUNS ?= 20
ATTN_PROBE_WARMUP ?= 5

.PHONY: check test compile benchmark benchmark-gpu benchmark-hf compare-hf-vllm attention-kernel-probe benchmark-awq render-gpu-report compare-prefix-cache compare-awq quality-smoke quality-smoke-bf16 quality-smoke-awq compare-quality serve-api serve-vllm serve-vllm-awq bench stream-bench \
	benchmark-prefix-cache experiment-baseline experiment-scheduler experiment-kv-cache compose-up compose-down compose-logs

check: compile test

benchmark:
	$(PYTHON) -m benchmarks.runner

test:
	$(PYTHON) -m pytest

compile:
	$(PYTHON) -m compileall -q app attention benchmarks engine experiments scripts tests visualization

serve-api:
	uvicorn app.main:app --host $(API_HOST) --port $(API_PORT)

serve-vllm:
	$(VLLM_BIN) serve $(MODEL_PATH) \
		--host $(VLLM_HOST) \
		--port $(VLLM_PORT) \
		--api-key $(API_KEY) \
		--gpu-memory-utilization 0.65 \
		--max-model-len 2048 \
		$(SERVER_EXTRA_ARGS)

serve-vllm-awq:
	$(VLLM_BIN) serve $(MODEL_AWQ_PATH) \
		--host $(VLLM_HOST) \
		--port $(VLLM_PORT) \
		--api-key $(API_KEY) \
		--gpu-memory-utilization 0.65 \
		--max-model-len 2048 \
		--quantization awq

benchmark-gpu:
	$(VLLM_PYTHON) scripts/gpu_benchmark.py \
		--model "$(MODEL_NAME)" \
		--api-key "$(API_KEY)" \
		--concurrency $(GPU_BENCH_CONCURRENCY) \
		--requests $(GPU_BENCH_REQUESTS) \
		--warmup $(GPU_BENCH_WARMUP) \
		--runs $(GPU_BENCH_RUNS) \
		--max-tokens $(GPU_BENCH_MAX_TOKENS) \
		--experiment-variant "$(GPU_BENCH_VARIANT)" \
		--server-args "$(GPU_BENCH_SERVER_ARGS)" \
		--output-dir $(GPU_BENCH_OUTPUT)

benchmark-awq:
	$(VLLM_PYTHON) scripts/gpu_benchmark.py \
		--model "$(MODEL_AWQ_PATH)" \
		--api-key "$(API_KEY)" \
		--concurrency $(GPU_BENCH_CONCURRENCY) \
		--requests $(GPU_BENCH_REQUESTS) \
		--warmup $(GPU_BENCH_WARMUP) \
		--runs $(GPU_BENCH_RUNS) \
		--max-tokens $(GPU_BENCH_MAX_TOKENS) \
		--experiment-variant "awq_int4" \
		--server-args "--quantization awq" \
		--output-dir $(GPU_AWQ_OUTPUT)

benchmark-hf:
	$(VLLM_PYTHON) scripts/hf_benchmark.py \
		--model "$(MODEL_PATH)" \
		--concurrency $(HF_BENCH_CONCURRENCY) \
		--requests $(HF_BENCH_REQUESTS) \
		--warmup $(HF_BENCH_WARMUP) \
		--runs $(HF_BENCH_RUNS) \
		--max-tokens $(HF_BENCH_MAX_TOKENS) \
		--dtype $(HF_BENCH_DTYPE) \
		--device $(HF_BENCH_DEVICE) \
		--output-dir $(HF_BENCH_OUTPUT)

compare-hf-vllm:
	$(VLLM_PYTHON) scripts/compare_hf_vllm.py \
		--hf reports/hf_benchmark \
		--vllm reports/gpu_benchmark

attention-kernel-probe:
	$(VLLM_PYTHON) scripts/attention_kernel_probe.py \
		--seq-lens $(ATTN_PROBE_SEQ_LENS) \
		--backends $(ATTN_PROBE_BACKENDS) \
		--dtype $(ATTN_PROBE_DTYPE) \
		--warmup $(ATTN_PROBE_WARMUP) \
		--runs $(ATTN_PROBE_RUNS) \
		--output-dir $(ATTN_PROBE_OUTPUT)

render-gpu-report:
	$(VLLM_PYTHON) scripts/render_gpu_report.py --input-dir $(GPU_BENCH_OUTPUT)

benchmark-prefix-cache:
	$(VLLM_PYTHON) scripts/gpu_benchmark.py \
		--model "$(MODEL_NAME)" \
		--api-key "$(API_KEY)" \
		--prompt-type long \
		--prompt-mode shared_prefix \
		--concurrency $(GPU_PREFIX_CONCURRENCY) \
		--requests $(GPU_PREFIX_REQUESTS) \
		--warmup $(GPU_PREFIX_WARMUP) \
		--runs $(GPU_PREFIX_RUNS) \
		--max-tokens $(GPU_PREFIX_MAX_TOKENS) \
		--experiment-variant "$(GPU_PREFIX_VARIANT)" \
		--server-args "$(GPU_PREFIX_SERVER_ARGS)" \
		--output-dir $(GPU_PREFIX_OUTPUT)

compare-prefix-cache:
	$(VLLM_PYTHON) scripts/compare_gpu_benchmarks.py \
		--before reports/gpu_prefix_cache_off \
		--after reports/gpu_prefix_cache_on

compare-awq:
	$(VLLM_PYTHON) scripts/compare_quantization.py \
		--bf16 reports/gpu_benchmark \
		--awq reports/gpu_awq

quality-smoke:
	$(VLLM_PYTHON) scripts/quality_smoke.py \
		--model "$(MODEL_NAME)" \
		--api-key "$(API_KEY)" \
		--max-tokens $(QUALITY_MAX_TOKENS) \
		--label "$(GPU_BENCH_VARIANT)" \
		--output-dir $(QUALITY_OUTPUT)

quality-smoke-bf16:
	$(MAKE) quality-smoke QUALITY_OUTPUT=reports/quality_smoke_bf16 MODEL_NAME="$(MODEL_PATH)" GPU_BENCH_VARIANT=bf16_baseline

quality-smoke-awq:
	$(MAKE) quality-smoke QUALITY_OUTPUT=reports/quality_smoke_awq MODEL_NAME="$(MODEL_AWQ_PATH)" GPU_BENCH_VARIANT=awq_int4

compare-quality:
	$(VLLM_PYTHON) scripts/compare_quality.py \
		--bf16 reports/quality_smoke_bf16 \
		--awq reports/quality_smoke_awq

bench:
	$(PYTHON) scripts/inference_bench.py --concurrency 1,2,4 --requests 10

stream-bench:
	$(PYTHON) scripts/stream_bench.py

experiment-baseline:
	$(PYTHON) -m experiments.runner experiments/configs/exp001_baseline.yaml

experiment-scheduler:
	$(PYTHON) -m experiments.runner experiments/configs/exp002_scheduler_sweep.yaml

experiment-kv-cache:
	$(PYTHON) -m experiments.runner experiments/configs/exp003_kv_cache_prefix.yaml

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f
