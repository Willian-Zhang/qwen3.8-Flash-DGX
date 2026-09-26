# Weight-load time investigation (2026-09-24)

Handoff notes. Goal: cut the ~9 min of "Loading weights" at every boot of `qwen38-flash`.

## TL;DR

- **~90% of weight loading is vLLM copying routed-expert tensors to the GPU one at a time straight
  from the memory-mapped checkpoint** (`expert_data.copy_(loaded_weight)` in `_load_w13` / `_load_w2`,
  `vllm/model_executor/layers/fused_moe/routed_experts.py`). ~149k tensors (800 KiB weights +
  100 KiB block scales); in a real boot that works out to ~3 ms each, in the benchmark 1.74 ms.
- **Not storage.** Local NVMe and NFS load at the same speed; iowait ~0.5%; pre-reading the file into
  the page cache first changes nothing.
- **The same copy from ordinary (anonymous) memory is 7–8× faster**: `.clone()` the mmap view first,
  then `copy_` → 0.23 ms/tensor (vs 1.74), bytes identical.
- **Fixed by patch 14** (`src/patch_moe_load_clone.py`, `VLLM_LOAD_CLONE=0` disables): clone before
  the H2D copy in the FusedMoE loader. First boot with it: main load 541 → 150 s, MTP 46 → 32 s,
  startup ~11 min → 4 min 32 s, smoke test unchanged.
- **Patches 15–18** (pread for small tensors, indexed expert-name matching, chunked embedding copy,
  MTP name prefilter; see "Patches 15–17" below): main load 150 → 35.5 s, MTP 32 → 1.2 s, startup
  4 min 32 s → **2 min 8 s**. Greedy probe identical; drafter counts identical with patch 18 on and off.
- **2026-09-26: the cause is unmapped pages, not file-backed ones.** The copy is slow when its
  source pages are not yet mapped in the process's page tables, cached or not; once touched through
  the mapping they copy in 0.021 ms. Touching 1 byte/page first ("prefault") alone takes the main
  load 443 → 26.5 s in a boot A/B. See "Mechanism corrected, prefault boot A/B" below.

## State at handoff

- Image: patches 1–18 (`qwen38-flash-dgx:latest`), boot 2 min 8 s.
- Serving `nvidia/Qwen3.8-Flash-Next-NVFP4`, `MODE=hybrid` (snapshot `fc694b5…-fp8hybrid`), YaRN 500k,
  MTP=2, `COMPILE_CACHE=qwen38` (docker volumes, reused: init engine ~119 s → ~33 s), port 8000.
  Settings: `systemd/qwen38-flash.env`; unit: `systemd/qwen38-flash.service`.
- Weights are served from the kc3000 NFS share first (`HF_HUB_DIRS`, NFS over RDMA, mounted `ro` at
  `/mnt/kc3000-nfs`), local `~/.cache/huggingface/hub` is the boot-time fallback (full copy kept).
- Branch `spark-service` = `upstream/main` (blazux, 5be6637) + the systemd service, `HF_HUB_DIRS`
  and patches 14–18 (pushed). Upstream: patch 14 merged as blazux#33; 15–18, ported to
  `Dockerfile.v0.30` and boot-tested there, are blazux#34 (branch `load-patches-15-18`, worktree
  `~/run/qwen-pr-load`). vLLM: vllm-project/vllm#58720 (patch 16, merged 09-26 in a rewritten
  form), issue vllm-project/vllm#58726 (the mmap H2D path; prefault proposal posted 09-26, see the
  last section). The LMCache experiment lives on its own branch, `lmcache`.
- `/mnt/models/gb10` (Synology NFS, 10 GbE) is **the backup mount — never delete anything there**.
  It holds a verified RadixArk checkpoint + hybrid (restored and sha-checked 2026-09-23); it is
  deliberately not in `HF_HUB_DIRS`.
- The kc3000 server export allows rw for 10.10.9.2 only (`/etc/exports`, backup `/etc/exports.bak`
  on the server); the client mount stays `ro`. A temporary rw mount (`/mnt/kc3000-rw`) is how files
  were copied onto the share.

## Baseline (same checkpoint and recipe)

| boot | weights from | "Loading weights" main + MTP | init engine (compile) | total startup |
|---|---|---|---|---|
| 09-23 08:55 | local NVMe | 499 + 60 s | 124 s (35.6) | 12 min 40 s |
| 09-23 09:51 | NFS | 530 + 46 s | 121 s (35.5) | 12 min 44 s |
| 09-23 10:48 | NFS, cache populating | 450 + 46 s | 119 s (34.8) | 11 min 23 s |
| 09-23 11:45 | NFS, compile cache reused | 518 + 46 s | **33 s (0.6)** | 10 min 53 s |
| 09-24 07:04 | NFS, profiled (py-spy) | 541 + 46 s | 35 s (0.6) | ~11 min |
| 09-24 07:34 | NFS, **patch 14** | **150 + 32 s** | 33 s (0.6) | **4 min 32 s** |
| 09-24 12:54 | NFS, **patches 14–17** | **35 + 12 s** | 34 s (2.0) | **2 min 13 s** |
| 09-24 13:41 | NFS, **patches 14–18** | **35 + 1.2 s** | 34 s (3.5) | **2 min 8 s** |

Weight loading varies ±40 s boot to boot (upstream saw 464–554 s too). Measure the patch against
the `Loading weights took` lines, not the total.

## How it was measured

1. **netdata** (runs on the host, `http://localhost:19999/api/v1/data?chart=...`) during the 11:45 boot:
   CPU ~5% user + ~1.5% system of 20 cores (= one busy core), iowait ~0.5%, ~150k minor page
   faults/s, RAM "used" jumps to ~88 GiB in the first 90 s (weights allocated up front), then 2–5 GiB
   free. RDMA NFS traffic is invisible to the interface counters.
2. **Tensor count** from `model.safetensors.index.json`: 299,845 tensors, 297,984 of them per-expert
   (48 layers × 512 experts × 3 proj × {weight U8 (640,1280)/(2560,320) = 800 KiB,
   weight_scale F8 100 KiB, weight_scale_2 F32 scalar, input_scale F32 scalar}, plus the MTP
   layer's 1,536 fp8 weights + `weight_scale_inv`). RadixArk: 296,775. Only weight + block scale go
   through `_load_w13` / `_load_w2`: **148,992 tensors**. The ~147k scalars take
   `_load_per_tensor_weight_scale` (one indexed assignment each, from the same mmap views).
3. **py-spy** in the container during loading. Needs `DOCKER_EXTRA=--cap-add=SYS_PTRACE` in the env
   file for that boot (removed again afterwards; `serve.sh` passes `DOCKER_EXTRA` to `docker run`),
   then `docker exec qwen38-flash pip install py-spy`, EngineCore pid via `pgrep -f EngineCore` in
   the container. `--native` cannot be combined with `--nonblocking`. Artifacts:
   `~/run/qwen-load-profile/` (`py.raw`, `native.raw` collapsed stacks, `dump_native.txt`).
4. **Micro-benchmarks** on real shards (`tools/bench_moe_load.py`), GPU only available with the
   service stopped (with vLLM up, CUDA sees ~1 GiB free and the benchmark OOMs).

## Findings

### Profile (Python samples, 60 s during loading, 99% of samples in `load_weights`)

| self time | frame |
|---|---|
| 59.6% | `_load_w13` (routed_experts.py:525) → `expert_data.copy_(loaded_weight)` |
| 29.6% | `_load_w2` (routed_experts.py:560) → `expert_data.copy_(loaded_weight)` |
| 4.3% + 1.0% | `FusedMoE.load_weights` (routed_experts.py:901/900) |
| ~4% | linear layers: `load_merged_column_weight` (parameter.py:176), `load_row_parallel_weight` (:230), `load_qkv_weight` (:201), `linear.py:377` |

Native samples: the copy is `at::native::copy_kernel_cuda` → `cudaMemcpyAsync` → `cuMemcpyHtoDAsync_v2`
from pageable memory; leaf time is spinning inside `libcuda` and `[vdso]` (clock polling), i.e. the
CPU waits for each small synchronous copy. `maybe_fuse_shared_experts` (models/utils.py:463) shows
~10% in the native profile.

### CPU side was not the main cost before patch 14 (`get_tensor` + copy into host memory, 800 KiB tensors)

With patch 14 in, this read is what is left of the expert cost; see "The rest of the boot".

| | NFS | local NVMe |
|---|---|---|
| mmap view → copy into ordinary memory, file not cached | 0.37 ms | 0.73 ms |
| same, file cached | 0.045 ms | 0.029 ms |
| plain `pread()` | 0.04 ms | 0.07 ms |

### GPU copy: whole 9 GiB shard, cold, fresh process per variant, untouched shard per run

| variant | ms/tensor | × 149k expert tensors |
|---|---|---|
| **A** vLLM today: mmap view → `param.copy_()` | 1.74 | ~259 s |
| **E** read the shard into the page cache first (8 threads, 0.7 s), then A | 1.86–1.89 | ~280 s — no help |
| **F** `.clone()` the mmap view, then the same `copy_` | **0.23** | **~34 s** |
| **B** reused pinned bounce buffer, one H2D per tensor | 0.22–0.26 | ~33–39 s |
| **C** pinned staging per 3072 tensors, 1 H2D + GPU scatter | 0.25 | ~37 s |

All variants verified byte-identical. "Cold" = client page cache dropped with
`posix_fadvise(DONTNEED)`; the NFS server may still have had the data in RAM, the same as in a real
boot. The benchmark does **not** reproduce the boot's absolute rate. ~89% of a 541 s load is ~480 s
for 149k tensors, ~3.2 ms each, about 2× variant A. That could be memory pressure (2–5 GiB free
during loading), or `copy_` seeing narrowed views rather than whole tensors; neither is verified.
The ratio carried over: 541 → 150 s in the boot.

**Superseded 2026-09-26** (see "Mechanism corrected, prefault boot A/B"): the slow case is source
pages not mapped in the process, not file-backed pages. Original conclusion, kept for the record:

**Conclusion:** the per-tensor pageable H2D copy is slow *when the source is a file-backed mmap page*,
cached or not; from anonymous memory it is fast. Why is unverified — plausible: the CUDA driver's
pageable-copy path on GB10 (ATS/HMM, unified memory) pins or walks file-backed pages per copy far
more expensively than anonymous ones. A one-line `.clone()` is as good as pinned staging.

### Ruled out

- Faster storage / a second 100 G cable: loading uses ~1% of the link; NVMe and NFS are identical.
- `--safetensors-load-strategy prefetch`: vLLM auto-skips it (121 GiB > free RAM), and E shows a warm
  page cache does not help anyway.
- `eager` and `enable_multithread_load`: both read whole files into RAM; one file is the 53.7 GB
  PLE shard (`model-fp8-mtp-ple.safetensors`) and only 2–5 GiB is free during loading → OOM.
- Other load formats (fastsafetensors would stage the 50 GiB PLE file on the GPU; runai streamer still
  feeds the same per-tensor loader) — not tried, not promising.

## Patch 14 (implemented 2026-09-24)

`src/patch_moe_load_clone.py`: the final `expert_data.copy_(loaded_weight)` of `_load_w13` and
`_load_w2` becomes `copy_(_qwen38_h2d_src(expert_data, loaded_weight))`, which clones a CPU,
non-pinned source when the destination is not on the CPU. Gate `VLLM_LOAD_CLONE` (default on, read
at import). Both weights and block scales go through these two sites, the MTP drafter's fp8 experts
too (patch 11's shim has no loader of its own; MTP load 46 → 32 s). Memory cost: one transient
≤ 800 KiB clone per call.

The remaining 150 + 32 s are broken down in the next section.

- Upstream: blazux/qwen3.8-Flash-DGX#33 (branch `moe-load-clone`, both Dockerfiles, a condensed
  write-up in `docs/HOW-IT-WORKS.md`). vLLM itself not yet: it affects every per-expert ModelOpt
  NVFP4 checkpoint on GB10.

### Validation (07:34 boot)

- In-image check: both call sites rewritten, gate on, 200 real expert tensors byte-identical on
  the GPU with the gate on and off.
- Boot: 74.9 GiB model memory (unchanged), KV pool 718,181 tokens (within the usual 625k–760k).
- `scripts/smoke-test.sh localhost:8000`: coherent, effort alias OK, 2,034 tok/s prefill at 8k,
  prefix-cache hit, deterministic, 35.0 tok/s decode — same as before.
- Rollback image: `qwen38-flash-dgx:pre-patch14`.

## The rest of the boot (profiled 2026-09-24 12:10, patch 14 on)

One boot recorded with `tools/profile_boot.sh`: py-spy on every process in the container, 10 s
slices from container start to "Application startup complete", 100 Hz, idle threads included.
Summarized with `tools/pyspy_slices.py`. Raw data: `~/run/qwen-load-profile/boot/NN-HHMMSS.txt`.
The run needs `DOCKER_EXTRA="--cap-add=SYS_PTRACE -v /home/willian/run/qwen-load-profile:/prof"` in
the env file for that boot, with py-spy staged once into `~/run/qwen-load-profile/pyspy` (see the
script header). The boot matched the unprofiled one: 149 + 32 s load, ready in 4 min 32 s.

### Timeline (seconds from container start, 272 s to ready)

| phase | s | notes |
|---|---|---|
| APIServer imports + config, then EngineCore spawn + imports | 25 | plain Python imports; the multimodal registry check is ~5 s of it |
| engine init, model construction | 7 | |
| main `Loading weights` | 149 | table below |
| MTP `Loading weights` | 32 | table below |
| init engine: memory profile (dummy forward ~8 s, vision encoder profile ~6 s), KV, graphs | 34 | compile cache already reused |
| APIServer: **multimodal warmup** (building the Qwen3-VL image processor) | 15 | EngineCore idle meanwhile |
| rest of API startup | 6 | |

### Main load, 149 s (EngineCore MainThread samples under `nvidia/model.py`)

| s (≈) | frame | cause |
|---|---|---|
| 45 | `_qwen38_h2d_src` (patch 14's `.clone()`) | page-faulting cold file pages through the mmap view, ~1.4 GiB/s |
| 26 | `_load_w13` / `_load_w2` `copy_` + the rest of the loader chain | the per-tensor H2D copy itself and its Python overhead |
| 25 | `FusedMoE.load_weights` lines 900/901 | **pure Python**: every tensor is substring-matched against all 1,536 expert-mapping entries (measured in isolation: 103 µs/tensor × 297k = **30 s**). Same loop on vLLM `main`. |
| 25 | linear layers (`parameter.py:154/176/201/230`, `linear.py:377`) | `param.copy_` straight from the mmap view, i.e. the same file-backed H2D slow path patch 14 fixed for the experts |
| 16 | `vocab_parallel_embedding.py:487` | `embed_tokens` + `lm_head`, 2 × 1.2 GiB H2D straight from the mmap (~150 MB/s) |
| 13–15 | `prewarm` (`PREWARM=1`) | streams the 47.7 GiB PLE table; afterwards only **12.7%** of the PLE file was still resident (mincore) — the 70 GiB weight stream that follows evicts it (inference; confirm with a `PREWARM=0` boot) |

### MTP load, 32 s (under `nvidia/mtp.py`)

| s (≈) | cause |
|---|---|
| 15 | the drafter loads its own `embed_tokens` + `lm_head` from the checkpoint (same slow H2D), then `load_eagle_model` (`v1/worker/gpu/spec_decode/eagle/utils.py`) deletes them and shares the target's (the model sets no `has_own_*` flag, so it always shares) — **wasted** |
| 13 | `get_all_weights` walks all 11 files / 299,845 tensors to keep 3,101 MTP ones (`remap_weight_names` filters *after* `get_tensor`); a cold `get_tensor` is ~50 µs (5.5 µs warm), and the main load has evicted the pages by then. 3,072 of the MTP tensors are in `model-fp8-mtp-ple.safetensors`, 29 in `model-00009`/`00010` |
| ~4 | the MTP experts and layers themselves |

### Read-path measurements (CPU only, client cache dropped, byte-identical)

| read path for the weight + block-scale tensors | GiB/s | ms/tensor |
|---|---|---|
| mmap `get_tensor` + `.clone()` (patch 14 today), cold | 1.39 | 0.339 |
| **`pread` each tensor straight into a fresh CPU tensor**, cold | **3.15** | **0.138** |
| sequential 16 MiB `pread` of the region, cold | 4.30 | — |
| mmap `get_tensor` + `.clone()`, region already cached | 4.64 | 0.147 |

(Shard 00005, alternating 256 MiB regions between the two methods. A first try on shard 00007 with
~1 GiB regions measured 0.47 GiB/s for the mmap path; the rate depends on how the faults line up.)

In the loader's real order (`f.keys()`, by name), a whole shard, cold client cache verified with
`mincore` (0.3% resident before):

| read path | shard 00006 (8.97 GiB, 39,597 tensors) | shard 00002 |
|---|---|---|
| mmap `get_tensor` + `.clone()` | 4.13 s = 2.17 GiB/s | 3.95 s = 2.26 GiB/s |
| `pread` per tensor | **0.82 s = 10.9 GiB/s** | 0.82 s = 10.9 GiB/s |

Caveat: `POSIX_FADV_DONTNEED` cannot drop pages that are still mapped, so a cold measurement
needs the file unmapped everywhere (vLLM maps none after boot, checked in `/proc/*/maps`). The
kc3000 server has the files in RAM, the same as in a real boot. A clone of an 800 KiB tensor
that is already in ordinary memory costs 9.8 µs, so patch 14's clone is redundant after (a)
(~1.5 s in total); (a) tags the storage it allocates and patch 14 skips those.

### How these interact with the PLE offload

The PLE table (`model-fp8-mtp-ple.safetensors`, 128 tensors `ngram_embedding.shard_N` of 381 MiB,
47.7 GiB) is not loaded by vLLM's loader. `src/vllm_ple_mmap.py` finds the shards' offsets in the
safetensors header itself, maps them with its own `np.memmap` (`MADV_RANDOM`), and its
`load_weights` drops the tensors the loader hands it ("served from disk, never materialised").
At inference each step gathers rows from that memmap: page-cache hits are cheap, misses go to NFS.

- a: must never read the shards. They are above the 64 MiB cap, and the patch also skips
  `ngram_embedding.shard_*` by name, so another checkpoint layout with small shards cannot
  make it read 47.7 GiB into RAM.
- e: sits in `VocabParallelEmbedding.weight_loader`; the PLE layer is a placeholder
  (`_MmapNgramEmbedding`, not a `VocabParallelEmbedding`), and its shards never reach a
  weight loader.
- b, c, d, g: nothing to do with the table.
- **What does interact is the page cache**, where the PLE's hot rows live. The weight load streams
  ~70 GiB of checkpoint pages through it; they are useless once the weights are on the GPU. That
  happens with mmap or with `pread` alike, so a neither helps nor hurts here.

### Correction: f (`PREWARM=0`) withdrawn

The 12.7% PLE residency above was measured *after* benchmarks that had read several GiB of other
shards, so it does not show that the weight load evicts the prewarmed table. Even if it does, the
right fix is not to drop the prewarm but to stop the loader from evicting it: after reading a
file's tensors, `posix_fadvise(DONTNEED)` the ranges it read (never the PLE table's), so the
prewarmed rows survive (f' below). Upper bound: ~18 GiB of page cache is left after boot, i.e.
at most ~38% of the table. Measure first: PLE residency right at "Application startup complete",
before any request, with and without f'.

### c and d in plain words

The MTP drafter is loaded as a second model after the main one, by the same loader, from the
same checkpoint.

- **c**: the checkpoint has one `embed_tokens` and one `lm_head` (1.2 GiB each). The main model
  loads them. Then the drafter loads its *own* copy of both, and immediately afterwards vLLM
  (`load_eagle_model`) deletes the drafter's copies and points the drafter at the main model's.
  So ~2.4 GiB is copied to the GPU for nothing (~15 s today, ~2–3 s once e is in). The fix: don't
  load them in the drafter (keep the empty parameters, vLLM replaces them anyway).
- **d**: to find its ~3,100 tensors, the drafter's loader opens every file and creates all
  299,845 tensors, then throws away everything that isn't an MTP weight by name. Creating a
  tensor costs ~50 µs when its pages are cold, so ~13 s go to tensors that are discarded. The
  fix: check the name *before* creating the tensor (vLLM already has a hook for that,
  `should_skip_weight`), or open only the 3 files that hold MTP weights.
- **a without d**: with a, the drafter's pass *reads* every small tensor (~70 GiB, all the main
  model's) before discarding it, instead of only creating views. At 10.9 GiB/s that is ~7 s,
  no slower than today's ~13 s, but all of it wasted I/O and page-cache churn. d removes it.

### Fix candidates

| # | change | est. saving | risk / notes |
|---|---|---|---|
| a | **`pread` loader**: in `safetensors_weights_iterator`, return tensors ≤ 64 MiB as `pread` copies into ordinary memory instead of mmap views (bigger ones, e.g. the PLE table and embeddings, stay views). Covers experts, scales, linear layers, MTP. Patch 14's clone becomes a no-op for them (the source is no longer file-backed). | ~60–70 s (expert clone 45 → ~18, linear 25 → ~5, plus scales) | one site; transient memory one tensor; the scalars also stop page-faulting |
| b | **MoE name-match index**: build `{weight_name: entries}` once per layer and look up by the `experts.N.proj.` fragment of the name instead of scanning 1,536 entries | ~25–30 s | pure Python, byte-identical; worth upstreaming to vLLM too |
| c | **MTP: skip the drafter's `embed_tokens` / `lm_head`** in `Qwen3_8FlashNextMTP.load_weights` when the target shares them (always, here) | ~15 s | must keep the parameters so the shape check / sharing still works; they are replaced right after |
| d | **MTP: skip non-MTP names before `get_tensor`** (filter by name in the iterator, or pass only the files the index maps MTP weights to) | ~12 s | vLLM already has a hook: `should_skip_weight(name, ...)` right before `get_tensor` |
| e | **Large tensors (`embed_tokens`, `lm_head`)**: copy to the GPU in 64 MiB pieces, each cloned into ordinary memory first | ~12 s | only 2 tensors (4 with the MTP pass); bounded memory |
| ~~f~~ | ~~`PREWARM=0`~~ — withdrawn, see the correction above | | |
| f' | **loader drops its own pages** (`DONTNEED` on the ranges read, not the PLE table) so the prewarm survives | 0 s at boot; faster first requests | measure PLE residency after boot first |
| g | **`--language-model-only`** (config only; the model supports it: vision tower becomes a `StageMissingLayer`, `visual.` weights skipped) | ~15 s API warmup + ~6 s encoder profile + a little memory | drops image input; the QSA fused rope path is also enabled with it (`qsa.py:298`, `text_only`) — check the smoke test and determinism |

a + b + c + d + e + g together would take the boot from ~4 min 32 s to roughly 2–2.5 min (estimate).
The weight-load parts (a–e, f') are code patches in the repo's style; g is an env-file change.

**Implemented: a, b, e** as patches 15, 16, 17, and **d** as patch 18 (below). c, f', g not yet.

## Patches 15–17 (implemented 2026-09-24)

| patch | file | gate (default on) | what |
|---|---|---|---|
| 15 (a) | `src/patch_load_pread.py` | `VLLM_LOAD_PREAD=0` disables | `safetensors_weights_iterator`: tensors ≤ 64 MiB are `pread` into ordinary memory, storage tagged `_qwen38_anon`; > 64 MiB and `ngram_embedding.shard_*` stay mmap views. Wraps patch 14's `_qwen38_h2d_src` to skip the clone for tagged tensors (needs 14). |
| 16 (b) | `src/patch_moe_name_index.py` | `VLLM_MOE_NAME_INDEX=0` | `RoutedExperts.load_weights` iterates only the mapping entries whose `weight_name` occurs in the tensor name (lookup at each `experts.` position, `{len: {name: [idx]}}`), original order, fused tensors cut at the first consecutive run like the original `break`. Full scan if an entry does not start with `experts.`. |
| 17 (e) | `src/patch_embed_chunked_copy.py` | `VLLM_LOAD_EMBED_CHUNK=0` | `VocabParallelEmbedding.weight_loader`: CPU → GPU in 64 MiB row blocks, each cloned into ordinary memory first; direct copy for tagged, pinned, 0-d or shape-mismatched sources. |

| 18 (d) | `src/patch_mtp_name_prefilter.py` | `VLLM_MTP_NAME_PREFILTER=0` | wraps `weight_utils.should_skip_weight` (the hook the iterator calls before reading each tensor) with an optional keep-filter; `Qwen3_8FlashNextMTP.load_weights` sets it to `_remap_mtp_weight_name(n) is not None` while it loads, unless the model has secondary weight sources (other name prefixes). |

Preview `Dockerfile` only (not `Dockerfile.v0.29`, not in the upstream PR yet).

### Validation (12:54 boot)

- CPU tests in the image: `src/test_moe_name_index_cpu.py` — 22,516 name/config cases (512 experts,
  EPLB redundant experts, w1/w2/w3 names, LoRA prefix, fused and per-expert-fused names) identical
  to the original loop, 73.5 → 1.1 µs per tensor. `src/test_load_patches_cpu.py` on
  `model-00009`, `model-00010` and `model-fp8-mtp-ple` — every small tensor byte-identical to the
  stock iterator; `embed_tokens`/`lm_head` stay views; all 128 PLE shards stay unread views.
  Chunked copy == plain copy for uneven row counts. Patch 14 wrapper: tagged → no clone,
  untagged → clone.
- Boot: main load **149 → 35 s**, MTP **32 → 12 s**, "Model loading took" 191 → 56 s, startup
  **4 min 32 s → 2 min 13 s**. 74.9 GiB model memory, KV pool 721,212 tokens. Loader log: per
  pass, `model-00009` 9,044 tensors read / 2 views (embeddings), PLE file 3,073 read / 128 views.
- `scripts/greedy-probe.sh post-p15-solo` vs `pre-p15` (taken on the patch 14 boot): all 5 prompts
  identical in text and first-token logprobs. `scripts/smoke-test.sh`: deterministic, prefix-cache
  hit, 36.1 tok/s decode (35.0 before). No clean cold-prefill number from this boot (the smoke
  prompt was already prefix-cached by then).
- **Pitfall hit:** the first smoke test and probe were run *concurrently* — `Running: 2 reqs` in
  the engine log — which changes batch shapes: 2–3 of 5 probe texts flipped (even between two
  probes in the same boot), "logprobs identical: NO", 1,437 tok/s prefill, 27.5 tok/s decode. Run
  them one after the other.
- PLE residency at "Application startup complete", before any request: **8.5%** of the 50 GiB file
  (clean measurement this time). So `PREWARM=1`'s 47.7 GiB stream is almost entirely evicted by
  the time the server is up — the case for f'.
- Rollback images: `qwen38-flash-dgx:pre-patch15` (patch 14 only), `:pre-patch14`.

### Validation of patch 18 (13:41 boot)

- CPU test `src/test_mtp_prefilter_cpu.py` on the whole snapshot: inside the filter the iterator
  yields exactly the index names `_remap_mtp_weight_name` maps (3,103: 3,101 MTP + `embed_tokens`
  + `lm_head`), 296,742 skipped unread, 2.7 s for the 11 files; filter off again afterwards.
- Boot: MTP load **12 → 1.21 s**, log `MTP name prefilter: 3103 tensors kept, 296742 skipped
  before reading`; main 35.5 s; "Model loading took" 56 → 43 s; startup **2 min 8 s**; KV pool
  630,303 tokens (inside the usual 625k–760k boot-to-boot range).
- Greedy probe `post-p18` vs `pre-p18` (patches 14–17 boot): all 5 texts and first-token logprobs
  identical. Smoke test: deterministic, prefix-cache hit, 35.4 tok/s decode, 1,785 tok/s cold
  prefill (single sample; cold prefill depends on the PLE page cache, 2,034 on the patch 14 boot).
- Drafter check (the greedy probe cannot catch a bad drafter: the target verifies every token):
  `vllm:spec_decode_num_{draft,accepted}_tokens_total` around one solo probe. This boot, three runs:
  **1,208 drafted / 864 accepted every time**. Previous boot (patches 14–17): 1,210 / 863. By
  construction the drafter gets the same tensors in the same order (the filter only removes names
  it would drop), and a missing tensor would collapse acceptance, not move it by one token. The
  cross-boot repeatability of the drafter was never measured, so the 2-token difference is not
  attributed yet; an A/B boot with `VLLM_MTP_NAME_PREFILTER=0` on the same image would settle it.
- **A/B boot 09-25 11:01** (same image, `DOCKER_EXTRA="-e VLLM_MTP_NAME_PREFILTER=0"`, removed
  afterwards): drafter **1,208 / 864**, identical to the prefilter-on boot; greedy texts and
  logprobs identical. So patch 18 does not change the drafter; the 1,210 / 863 of the patch 14–17
  boot is a between-boots difference (likely the drafter's compiled graph, rebuilt once when
  patch 18 changed `mtp.py`; not verified). MTP load with the prefilter off: 7.0 s, on: 1.2 s.
- Rollback image: `qwen38-flash-dgx:pre-patch18` (patches 14–17).

### v0.30 port (09-25, image `qwen38-flash-dgx:v0.30-p18`, weights on local NVMe)

The scripts apply to v0.30.0 unchanged, except 18: `mtp.py` moved to `qwen4_exp`, and its
`load_weights` passes `mapper=mapper` (drops names only after the remap, so the kept set is the same).
Same image, A/B via the four switches, own compile cache (`COMPILE_CACHE=qwen38v030`):

| v0.30, local NVMe | patch 14 only | patches 14–18 (2 boots) |
|---|---|---|
| main load | 189.5 s | 97.4 / 99.5 s |
| MTP load | 48.7 s | 1.2 / 1.2 s |
| startup | 5 min 36 s | 3 min 17 s / 3 min 9 s |
| KV pool | 700,000 | 543,939 / 539,393 |

Greedy probe 5/5 texts and first-token logprobs identical between the two, drafter 872 / 592 on both,
smoke test green. The preview image from the same local NVMe: 97.2 + 2.5 s, 2 min 57 s, so local
NVMe (~1.46 GiB/s single stream) is what bounds the main load there; from NFS it is 33–35 s. The KV
pool difference is unexplained (one unpatched boot; preview boots range 622k–760k).

The new timeline: container start → EngineCore init 23 s, model construction + prewarm start 6 s,
main load 35 s (the PLE prewarm runs inside it), MTP load 12 s, init engine 34 s, API
multimodal warmup 14 s, rest ~9 s. After patch 18 the MTP load is 1.2 s. Remaining candidates: c
(< 1 s now that e and d are in), f' (page cache, first-request speed), g (~20 s, config).

## Mechanism corrected, prefault boot A/B (2026-09-26)

Prompted by hclsys on vllm#58726: on his GB10, cached pages copied fast (0.027 ms/tensor).

**Micro-benchmark** (`.cache/copy-recheck/`, git-ignored: `bench_copy_recheck.py`, `run.sh`,
`RESULTS.md`). Shard 00006 (19,714 expert tensors, 8.46 GiB), NFS and local NVMe, service stopped,
one fresh root container per run, residency checked with `cachestat` and `mincore`, two passes.
ms per tensor, whole per-tensor op:

| pages before the loop | A: view → `copy_` | F: clone | P: touch 1 B/page | M: `MADV_POPULATE_READ` |
|---|---|---|---|---|
| not cached | 1.7–3.0 | 0.24–0.62 | 0.08–0.64 | 0.21–1.11 |
| cached (`preadv`, = E) | 1.9–2.6 | 0.11–0.23 | 0.04–0.17 | 0.60–0.90 |
| cached and mapped (touched through the same mmap) | **0.021** | | | |

- The slow case is source pages that are not mapped in the process's page tables, cached or not.
  Mapped pages copy ~100× faster. File-backed vs anonymous memory does not matter; the clone helps
  because its output is freshly written, mapped memory. E was a valid control (p1 reproduced it:
  1.93 ms).
- P was fastest or tied with the clone in every cell and allocates nothing. `MADV_POPULATE_READ` did
  not help; not understood.
- In a real boot nothing has touched the checkpoint through safetensors' mapping, so on GB10 the
  copy is always slow, warm cache or not.
- The 09-25 `mincore` anomaly is explained: as a non-root user on the root-owned local blobs the
  kernel reports every page resident (`mincore` = 1.000 right after `DONTNEED`) and `cachestat`
  returns EPERM. Measure as root (containers) or as the file's owner.

**Boot A/B** (`.cache/boot-ab/`, git-ignored: `ab.sh`, `logs/`, `RESULTS.md`). Weights from NFS,
client cache dropped before every boot, boots interleaved. Only the loader changed: a bind-mounted
`weight_utils.py` with a `VLLM_LOAD_PREFAULT` hook (no-op when unset) plus the patch 14/15
switches. Patches 16–18 on in every boot.

| config | main load | MTP load | start → `/health` | swapped out before the KV profile | KV cache |
|---|---|---|---|---|---|
| base: patch 15 `pread` + patch 14 clone (3 boots) | 21.9–22.5 s | 1.2 s | 103–104 s | 0.01–0.24 GiB | 16.9–17.0 GiB |
| prefault views ≤ 64 MiB, `pread` + clone off (3 boots) | 26.1–26.7 s | 0.9–1.0 s | 118–119 s | 2.8–2.9 GiB | 19.4–19.5 GiB |
| all three off (1 boot) | 443 s | 16.5 s | 549 s | 5.2 GiB | 19.1 GiB |

Greedy probe texts and first-token logprobs identical in all 7 boots. `pread` is ~4.3 s faster on
the main load (~15 s to ready). The prefault boots' bigger KV cache is the swap effect (see
`.cache/NOTES-2026-09-25.md`): they push ~2.9 GiB of other memory to swap during the load (netdata
`mem.swapio`), which vLLM counts as free. So it is not a memory saving, and `pread` is not worse
on memory. Production stays on `pread` + clone.

**Upstream status.** vllm#58720 merged 09-26 as `ad6817b68` (Michael Goin's rewrite of patch 16);
patch 16's script will need dropping once the base image includes it. vllm#58726: proposal posted
(prefault in `safetensors_weights_iterator`, asking maintainers about the gate and the size cap);
no reply yet. Next: measure prefault on the discrete-GPU machine, where the pageable copy is
staged through a CPU memcpy anyway.

## Rerunning the benchmark

Needs the service stopped (free GPU). One variant per fresh container, each on a shard no earlier
run touched in this session (otherwise it is not cold):

```sh
S=/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47-fp8hybrid
docker run --rm --gpus all --ipc=host --network none \
  -v /mnt/kc3000-nfs/cache/huggingface_hub:/hub:ro -v "$PWD/tools/bench_moe_load.py:/b.py:ro" \
  -e V=A -e F=$S/model-00006-of-00010.safetensors --entrypoint python3 qwen38-flash-dgx /b.py
```

`V` ∈ {A, E, F, B, C} (see the script header). Shards 00002–00009 hold the routed experts.
