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

## State at handoff

- Serving `nvidia/Qwen3.8-Flash-Next-NVFP4`, `MODE=hybrid` (snapshot `fc694b5…-fp8hybrid`), YaRN 500k,
  MTP=2, `COMPILE_CACHE=qwen38` (docker volumes, reused: init engine ~119 s → ~33 s), port 8000.
  Settings: `systemd/qwen38-flash.env`; unit: `systemd/qwen38-flash.service`.
- Weights are served from the kc3000 NFS share first (`HF_HUB_DIRS`, NFS over RDMA, mounted `ro` at
  `/mnt/kc3000-nfs`), local `~/.cache/huggingface/hub` is the boot-time fallback (full copy kept).
- Branch `spark-service` = `upstream/main` (blazux, 5be6637) + the systemd service, `HF_HUB_DIRS`
  and patch 14. The LMCache experiment lives on its own branch, `lmcache`.
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

### CPU side is not the cost (`get_tensor` + copy into host memory, 800 KiB tensors)

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

The remaining main load of 150 s is not profiled. Candidates: the ~147k scalar per-tensor scales
(`_load_per_tensor_weight_scale`, same mmap views, never measured), the linear layers,
`maybe_fuse_shared_experts`, and whatever part of the ~3 ms per-tensor boot cost the clone does not
remove. Options:
- Broader variant: clone in `safetensors_weights_iterator` (model_loader/weight_utils.py) for tensors
  below a size cap (e.g. 64 MiB), so the linear layers (~4–10%) and the MTP load (46 s) benefit too;
  the cap skips the huge PLE tensors, which the PLE mmap patch discards anyway.
- Re-profile one boot with py-spy to see what the remaining ~150 s is.
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
