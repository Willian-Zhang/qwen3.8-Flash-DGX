# vLLM v0.32 base — issues #58 and #59

The recipe uses `vllm/vllm-openai:v0.32.0`, labeled `qwen38.base=v0.32`.
`flash` and the download, preparation and serving scripts use the corresponding
`qwen38-flash-dgx:v0.32` image. Existing prepared NVFP4/hybrid checkpoints are
reusable; no new download or conversion is required. This migration keeps the
existing checkpoint, profile settings, MTP width, context limits and KV defaults.
FP4-Local, alternative backends and precision experiments are outside this port.

```bash
./flash setup
./flash serve default   # apply the same profile/overrides as before
./flash wait
./flash test
```

For source-only updates, rebuild explicitly: `docker build -t qwen38-flash-dgx:v0.32 .`.
`flash setup` reuses an image with the expected base label.

## Port decisions

- **Mamba prefix caching (patch 4):** v0.32 already seeds the worker's running
  state using `mamba_block_size`. Assert that implementation rather than applying
  the old replacement. The scheduler still uses `cache_config.block_size` in its
  prefill split; retain the local change to the scheduler's group LCM, preserving
  v0.32's new internal-checkpoint logic.
- **PLE mmap (patch 1):** forward the new `use_sequence_parallel` constructor
  argument. The mmap table is replicated, so token sharding uses `sp_shard`
  without reducing vocabulary contributions. GPU validation covers TP=1 only.
- **QSA projection fusion:** upstream combines QKV and the indexer into four
  shards. Existing hybrid blockwise-FP8 checkpoints and the local quantization
  dispatch require the original three-shard QKV mapping. Restore that mapping
  before constructing the initial QKV linear, so a BF16 indexer does not send
  FP8 Q/K/V to the unquantized path. Keep upstream's unfused
  path when either projection is blockwise FP8; keep fusion for other layouts.
  This avoids claiming the merged-GEMM speedup for hybrid mode. The upstream
  fused pre-indexer remains enabled independently.
- **Responses and tools (patches 12/13/19/21):** retain token identity, literal
  marker guards, schema repair and the Chat/Responses tool-name distinction.
  Rebase the nonstreaming error handler around v0.32's optional streamed-item
  reuse, and validate streaming calls before v0.32's parallel-call filter.
  Preserve upstream reasoning event names and streamed-item IDs.
- **MTP auxiliary output (patch 20):** adjust import context; keep target capture
  and clear the drafter's independent aux configuration.
- **KV:** native QSA FP8-e4m3 support remains in use, including upstream
  `--kv-cache-dtype-skip-layers` support. QSA still rejects NVFP4 KV; general SM12x
  support does not enable it for this backend. No new default KV dtype is chosen.
- **Startup:** keep the existing compile-cache mechanism. Kernel download/preload
  commands are not invoked automatically by this port.

The other PLE, FLA, top-k, draft-vocabulary and loading patches are retained.
`mamba_cache_mode=all` and cascade attention are not enabled by this recipe.
Historical v0.31 measurements remain in [VLLM-031.md](VLLM-031.md).

## Validation

The parser source manifest for this release is
[`tests/vllm032/PORT-MANIFEST.json`](../tests/vllm032/PORT-MANIFEST.json).
Run installed-code CPU QA in the built image with local tokenizer metadata:

```bash
docker run --rm --network none --runtime runc --entrypoint python3 \
  -e VLLM_TARGET_DEVICE=cpu -e CUDA_VISIBLE_DEVICES= \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e QWEN_TOKENIZER=/tokenizer \
  -v "$PWD/tests:/tests:ro" -v /path/to/tokenizer:/tokenizer:ro \
  qwen38-flash-dgx:v0.32 /tests/issue31_schema_arguments/run_qa.py \
  --manifest /tests/vllm032/PORT-MANIFEST.json
```

Run `/tests/issue31_safe_output/run_qa.py` and
`/tests/test_mtp_aux_output_cpu.py` separately with the same mounts.
`python3 tests/vllm032/test_hybrid_qsa.py` checks the fusion decision on a CPU.

### CPU results on GX10 (2026-10-21)

The official image builds successfully on aarch64, including the deterministic
top-k extension. Installed parser/Responses hashes match the release manifest.

- Schema producer integration: **11/11**; Chat compatibility: **10/10**.
- Author schema suite: **22/23**, retaining the previously documented dotted
  namespace limitation; no new failures or skips.
- Previous output regressions: **67/67 methods and 21/21 byte-fidelity cases**.
- MTP aux-output config: **4/4**; hybrid QSA selection: **4/4**; PLE constructor:
  **1/1**; real ModelOpt QKV mapping **2/2**; native reasoning events and streamed/final item IDs: **2/2**.
- Synthetic PLE gather and exact top-k checks pass. The MTP prefilter yields
  exactly 3,103 expected tensor names out of 299,845 and clears its filter.
- Official v0.32 parser-engine suite plus patch-12/13 tests: **5,913/5,914**
  on both the baseline (v0.32 plus patches 12/13) and the complete candidate.
  The sole failure is the existing malformed-opener/literal-marker preservation
  disagreement, `test_content_around_unpromoted_tool_block_stays_ordered`.

Raw CPU results and scope are preserved in
[`tests/vllm032/validation/`](../tests/vllm032/validation/).
These CPU checks use synthetic engine frames; they generate no model tokens.

### Clean migration review (2026-10-22)

The migration-only image builds on ARM64 with local cached layers, no build
network and `--fuzz=0`, without the FP4-Local loader or experimental backends.
The optional memory-diagnostic derivative also builds. All seven installed-code
CPU suites pass their expected outcomes; the schema author's namespace failure
remains recorded as **22/23**, with no additional failures or skips.

Hashes of **2,883 files** match the previously GPU-tested v0.32 image: all vLLM
Python sources, the top-level local `vllm_*.py` modules and the deterministic
top-k extension. This is source/extension identity, not a comparison of every
dependency or whole image. No new GPU inference was run and the production
container was not modified during this review. The complete upstream parser
suite was not rerun; its previous baseline/candidate failure is recorded above.

See the [reproduction instructions](../tests/vllm032/README.md),
[review summary](../tests/vllm032/validation/review-20261022/summary.json) and
[build log](../tests/vllm032/validation/review-20261022/build.txt).

## Live probes

The two stdlib-only probes make model-inference requests without executing
proposed tools. Run against an otherwise idle candidate:

```bash
python3 tests/vllm032/probe_serving.py --output /tmp/v032-protocol.json
python3 tests/vllm032/probe_long_context.py --tokens 499000 --output /tmp/v032-long-context.json
```

The long-context probe measures the formatted prompt through `/tokenize`,
checks an exact middle-of-document answer twice, compares greedy output, and
requires more than 90% cached input on replay. It tests one synthetic retrieval
case, rather than general long-context model quality. The full agentic
tournament is not rerun by these probes.

### FP8 comparison with production settings

One pass on the same GX10, NVIDIA hybrid checkpoint, YaRN 500k, MTP=2,
16 sequence slots, BF16 recurrent state, FP8-e4m3 KV fixed at 8 GiB, GPU
utilization 0.80, 8,192 batched tokens and PIECEWISE graphs:

| Measurement | v0.31 | v0.32 |
| --- | ---: | ---: |
| Code generation, median of 3, end-to-end output tok/s | 48.8 | 47.7 |
| 4 clients, aggregate output tok/s | 104.3 | 108.1 |
| 8 clients, aggregate output tok/s | 168.5 | 161.0 |
| Fresh prefill, 9,743 input tokens, tok/s | 3,133 | 3,159 |
| Fresh prefill, 38,743 input tokens, tok/s | 3,196 | 3,238 |
| Fresh prefill, 120,910 input tokens, tok/s | 3,145 | 3,179 |
| KV token capacity, fixed 8 GiB | 573,208 | 573,208 |

These samples show similar throughput, not an established speedup. Ollama and
LiteLLM remained running; external traffic was not isolated, CPU QA overlapped
part of the reference, and the OS page cache was not reset. Fresh prefill prompts
had a new prefix per size to prevent reuse from the earlier benchmark. The prefill
section of `qwen-v031-perf-clean.txt` reused cached prompts and must not be used
for a cold comparison; use `qwen-v031-prefill-fresh.txt` instead.

Live smoke checks pass for coherence, effort aliasing on `/v1/messages`, prefix
cache hits and identical first-token logprobs. Real Chat tool arguments and a
round trip with a fixed synthetic tool result pass. Full and streaming Responses
tool calls pass, including streamed/final item IDs. No proposed tool is executed.

The FP8 candidate became healthy in **3 min 36 s**. Its formatted long prompt
contained **498,994 input tokens**. The exact middle-of-document code was found
in both greedy requests with identical visible output: **184.60 s** fresh,
**3.61 s** replay, **496,048 cached tokens (99.4%)** on replay. This is one
synthetic retrieval case, not a broader quality score. Raw usage and timings
are in [`long-context.json`](../tests/vllm032/validation/live-fp8/long-context.json).

### Public default profile

`./flash serve default` also passes health, coherence, effort aliasing, prefix
cache reuse and identical first-token logprobs at **500k context**, MTP=2 and
8 sequence slots. KV uses BF16 (`KV_DTYPE=auto`); the recurrent-state dtype is
the model default, with no CLI override. CPU affinity is the script's automatic
performance-core selection (`5-9,15-19` on this GX10).

With `GPU_MEM=0.80` and automatic KV sizing, the boot reserves **16.43 GiB**,
providing **607,575 KV tokens**. The graph estimator remains enabled; no fixed
KV pool or allocator override is supplied. The API becomes healthy in
**3 min 30 s**. This is one startup observation, not a paired automatic-memory
comparison against v0.31 or a capacity guarantee across hosts and memory states.
The long 498,994-token retrieval above used the fixed FP8 pool; it was not
repeated under this BF16 profile.

[Default-profile settings and startup](../tests/vllm032/validation/live-default/summary.json),
[smoke output](../tests/vllm032/validation/live-default/smoke.txt).

Upstream references: [v0.32 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0),
[QSA implementation](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/models/qwen4_exp/nvidia/qsa.py),
[M-RoPE position limits](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/qwen3_vl.py).

After the initial port-validation tests, production was restored to its original
v0.31 container. This snapshot predates the optimization campaign below. Container
ID, image, command and environment match the pre-test snapshot; `/health` returns
200 and a fresh completion answers correctly.
[Restoration check](../tests/vllm032/validation/restoration.json).

## Independent recipe audit (issue #59)

Thanks to [@rp-tn](https://github.com/rp-tn) for testing the published recipe at
`ffa78a4` on the official v0.32 arm64 image and documenting the exact failures in
[issue #59](https://github.com/blazux/qwen3.8-Flash-DGX/issues/59). The audit agrees
with this port: recognize the native Mamba worker fix while retaining the
scheduler fix, adapt Responses error handling to streamed-item reuse, and refresh
MTP import context. Patch 21 depends on patch 19 and is rebased after it.
All patch commands retain `--fuzz=0`; relaxing patch context is not the fix.

Applying patches alone does not exercise the PLE constructor or the hybrid QSA
weight loader. The CPU and GPU checks above cover those additional contracts.

## Long-context MTP cache fix (issue #60)

[@FinSnow's report](https://github.com/blazux/qwen3.8-Flash-DGX/issues/60)
describes an EngineCore crash when a YaRN prompt crosses the native 262,144-token
boundary with MTP enabled. The matching upstream regression is
[vllm#62383](https://github.com/vllm-project/vllm/issues/62383): the default M-RoPE
cache can retain the native length even though the engine serves a larger window.
The MTP/CUDA-graph path can then index beyond the allocated table. Merely setting
the drafter's `max_model_len` does not resize this table.

Patch 22 adapts [@wtogami's proposed fix](https://github.com/vllm-project/vllm/pull/62409)
at head `892a8c6fe650ec76a53fa2ff26c9e67b3d6a9f07`. The cache follows the target's
served window, including when it is constructed for an MTP drafter. The backport
keeps `max_position_embeddings` unchanged: in this base, YaRN uses that value in
its frequency and correction-range calculations. Raising it before the original
constructor changes existing rows as well as cache capacity. Instead, the port
computes the original table, retains every row, and appends only missing positions
using the same frequencies and magnitude scale. Existing scaled and multimodal
headroom is preserved. No weights, sampling, profile, RoPE scaling parameters or
MTP width are changed.

The complete image builds with `--fuzz=0`. The new installed-code CPU suite gives
**10/10 passing methods**, compared with **5 failures on the unpatched reference**.
It checks both the actual 262k/500k bounds and the constructor/factory contracts,
including the target's window overriding the drafter's own native limit. Native
rows, BF16 rows, scaled rows and multimodal headroom are checked for exact
preservation. A runtime comparison covers 2,883 Python/source-extension files;
only `mrope.py` differs. This comparison does not assert complete image/dependency
identity.

The [boundary probe](../tests/vllm032/probe_mrope_boundary.py) makes real model
requests and can crash an unfixed engine; use a separately started test instance.
It tests approximately 260k, 263k, 336k and 499k input tokens, with fresh and
prefix-cached requests, exact retrieval, health checks and subsequent short
generations. It saves partial reports when a request fails. No proposed tools
are executed. Provenance and source hashes are recorded in
[`MROPE-MANIFEST.json`](../tests/vllm032/MROPE-MANIFEST.json).

GPU validation on GX10 uses the existing NVIDIA hybrid checkpoint, MTP=2,
YaRN 500k, FP8 KV fixed at 8 GiB, 16 sequence slots, BF16 recurrent state and
PIECEWISE graphs. Both arms use identical long-request payloads and package
versions (Torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0). The observed target
cache stays at **1,048,576 rows**; the draft cache grows from **262,144 to 500,000**.
A constructor logging wrapper is applied equally to the two test arms; it returns
the installed computation unchanged and is absent from the production rollout.

| Actual input tokens | Unpatched reference | Patch 22 |
| --- | --- | --- |
| 259,995 | Cold + replay pass | Cold + replay pass |
| 262,998 | Cold + replay pass | Cold + replay pass |
| 335,994 | Cold + replay pass | Cold + replay pass |
| 498,988 | Cold + replay pass | Cold + replay pass |

All eight long requests in each arm return the exact retrieved code. Replay
reuses more than 90% of the prefix, and each long request is followed by a
successful health check and short generation. **The reference did not reproduce
the reported EngineCore crash on this workload.** The undersized draft cache is
reproduced in CPU tests and observed in real model construction; patch 22 fixes
that capacity defect. Confirmation on the reporter's exact failing workload is
still useful. A separate MTP-off GPU control was not run because the reference
stayed healthy. These results establish no general long-context quality or speed
improvement.

The full eight-suite CPU rerun has no unexpected failures; the existing author's
dotted-namespace case remains the known schema failure. Raw CPU/GPU reports,
settings, source comparison and limitations are recorded in
[`validation/issue60-20261022/`](../tests/vllm032/validation/issue60-20261022/VALIDATION.json).

Existing v0.32 installations must rebuild the image explicitly and recreate the
container with the usual profile and overrides. `./flash setup` reuses a matching
base label and does not install source-only fixes into an existing image.

## Validation limits and rollback

This is a compatibility and maintenance update. These measurements establish no
net throughput gain, extra KV capacity at a fixed allocation, or improvement in
agentic quality. The complete v0.32 agentic tournament remains unrun. The known
schema namespace and malformed-opener parser failures are recorded above rather
than treated as passing tests. GPU validation covers one GX10 at TP=1.

The 500k synthetic retrieval validates the existing NVIDIA hybrid checkpoint and
the specified FP8 settings only. The draft-length override solves the startup
length check but does not propagate the target's YaRN RoPE configuration to MTP;
the test does not establish broad long-context quality.

Keep the old image and save the old container's launch settings before recreating
it. To roll back, use the v0.31 recipe from commit `ffa78a4` in a separate checkout
with `qwen38-flash-dgx:v0.31` and the original profile/overrides. The v0.32 launcher
checks its base label and deliberately does not launch a v0.31 image. Both recipes
reuse the existing checkpoint files; this port does not convert them.
