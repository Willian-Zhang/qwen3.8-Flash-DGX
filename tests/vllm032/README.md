# vLLM v0.32 compatibility validation

The migration keeps the existing checkpoint and serving-profile settings.
[VLLM-032.md](../../docs/VLLM-032.md) records the port decisions, GPU probes,
historical performance comparison and validation limits. The independent
[issue #59 audit](https://github.com/blazux/qwen3.8-Flash-DGX/issues/59) checks
patch application; the constructor and dispatch tests also check runtime
interfaces that patch application alone does not exercise.

`PORT-MANIFEST.json` records the nine installed parser/Responses file hashes.
The schema runner accepts `--manifest` so the v0.31 provenance stays intact.
The other suites check the mmap constructor, blockwise-FP8 QSA dispatch,
MTP auxiliary outputs, native Responses event names and final item IDs.

Build the recipe and run the installed-code CPU checks on the supported ARM64
host. Mount only `tests/`, not the repository's `src/`, so the compatibility
tests import the modules installed in the image. Provide local compatible Qwen
tokenizer metadata; no weights, GPU, network or running inference server are used.

```bash
docker build -t qwen38-flash-dgx:v0.32 .
docker run --rm --network none --runtime runc --entrypoint python3 \
  -e VLLM_TARGET_DEVICE=cpu -e CUDA_VISIBLE_DEVICES= \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e QWEN_TOKENIZER=/tokenizer \
  -v "$PWD/tests:/tests:ro" \
  -v /path/to/local/tokenizer:/tokenizer:ro \
  qwen38-flash-dgx:v0.32 /tests/vllm032/run_cpu_qa.py
```

The schema author's dotted-namespace case remains a known failure: **22/23**.
Its runner checks that the failure is exactly the documented one and rejects
additional failures, errors or skips. All other suites in this runner must pass.
The separate historical upstream parser-engine comparison is **5,913/5,914**
on both baseline and candidate; its malformed-opener failure remains recorded
in `validation/`. This runner does not rerun that full upstream suite.

The migration-only rebuild and seven-suite rerun on 2026-10-22 are recorded in
[`validation/review-20261022/summary.json`](validation/review-20261022/summary.json),
with per-suite diagnostics and build logs beside it. The earlier GPU results
remain in `validation/VALIDATION.json`; this CPU rerun did not generate tokens.

The live probes require a separately started candidate. They generate tokens
but do not execute proposed tools. A synthetic long-context retrieval and
protocol checks do not establish agentic tournament quality.
