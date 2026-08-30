# LMCache on Qwen3.8-Flash-Next / DGX Spark (GB10) — investigation record

*2026-08-29. Outcome: LMCache **cannot** cache this model yet (upstream limitation).
Consolation prize: vLLM's native GPU prefix caching now works on this model and is
enabled by default (`PREFIX_CACHE=1` in `scripts/serve.sh`).*

## Goal

Offload KV cache to NVMe with LMCache (https://docs.lmcache.ai) so long shared
prefixes survive across requests/restarts. On the GB10 the "CPU RAM" tier is the
**same 128 GB unified pool** the GPU uses, so RAM offload buys nothing — NVMe
(1 TB, ~234 GB free) is the only real cache tier.

## Environment

- Image `qwen38-flash-dgx` (vLLM `0.1.dev20073+g8e685d198`, torch 2.13 cu130,
  **lmcache 0.5.4 already bundled**), container `qwen38-flash`.
- Model is a hybrid: full attention + GDN/mamba + QSA sparse attention + a QSA
  indexer **circular-buffer** state. This hybridity drove every failure below.
- Boot takes ~13–14 min (76 GiB weights + 48 GiB PLE prewarm). Every config
  change costs a full cycle — check cheap things (imports, CLI flags, source in
  the image via `docker run --rm --entrypoint sh qwen38-flash-dgx -c 'grep …'`)
  **before** restarting.

## Timeline of failures → fixes

| # | Attempt | Failure | Fix |
|---|---|---|---|
| 1 | In-process `LMCacheConnectorV1`, env-var config (`LMCACHE_LOCAL_DISK=file:///…`) | `ValueError: Failed to promote local KV cache specs to one unified type` — any non-HMA connector turns off the hybrid KV cache manager, which hybrid SSM models require | Switch to MP mode: only `LMCacheMPConnector` subclasses `SupportsHMA` (verified by importing both in the image) |
| 2 | `lmcache server --gds-l1-path` (NVMe L1 via GDS) | `libcufile.so` missing from the image; `--gds-l1-backend ugds` needs a devdax char device | Small RAM L1 + `--l2-adapter '{"type":"fs_native","base_path":"/lmcache/l2","num_workers":8,"max_capacity_gb":150}'`. **Warning:** the failed GDS attempt left a root-owned 150 GB preallocated slab at `lmcache-disk/gds/` — delete via a container if it ever reappears |
| 3 | MP server on port 5555 | Port already taken on gx10 (something listens on 127.0.0.1:5555) | ZMQ on **5557**, HTTP mgmt on **8090** |
| 4 | MP connector, defaults | `MambaSpec with mamba_cache_mode='none' (only 'align' keeps reusable state snapshots)` | `--mamba-cache-mode align` |
| 5 | `--mamba-cache-mode align` alone | vLLM silently resets it: *"Mamba cache mode is set to 'none' when prefix caching is disabled"* (serve.sh had `--no-enable-prefix-caching`) | Also `--enable-prefix-caching` |
| 6 | chunk 256 | `LMCache chunk size 256 must be a multiple of engine group 0 tokens_per_block 1600` — mamba/attention page parity forces 1600-token attention blocks on this model | `--chunk-size 1600` |
| 7 | `lmcache_driven` (default) transfer | Server-side kernels reject the padded KV layout: `dim-0 padding … engine_kv_format=NL_X_NB_BS_NH_CS is not a supported dim-0-padded format` | `lmcache.mp.mp_transfer_mode=engine_driven` (workers do the copies; server accepted registration, pickle transport) |
| 8 | Everything boots, lookups reach the server (`lmcache_mp_lookup_requested_tokens_total` grows) | **Zero stores, ever.** Instrumented `GetStoreMetadata` (bind-mounted a patched `lmcache_mp_metadata.py` over the installed file) | See root cause below — not fixable by config |

## Root cause (the hard stop)

`STORE-DEBUG` trace from the instrumented connector:

```
tokens_per_block=[1600, 8, 1600, 1600, 1600, 1600]  alloc_lens={0:3, 1:1, 2:6, …}
alloc_tokens=8  staging=8  chunks=0
```

KV group 1 is the **QSA indexer circular buffer**: 8 tokens per block, exactly one
block per request, regardless of sequence length. LMCache computes the storable
prefix as `min(blocks × tokens_per_block)` **across all engine groups**
(`lmcache/integration/vllm/lmcache_mp_metadata.py::GetStoreMetadata`), so that
group caps it at 8 tokens forever → `chunks = 0` → no store is ever submitted.

- lmcache 0.5.4 group edits (`kv_cache_group_edits.py`) handle MAMBA
  (`mamba-unified-view`) and MLA, but have no concept of ring-buffer specs.
- Checked 0.5.5rc1 (newest on PyPI as of 2026-08-29): still no
  `CircularBufferSpec` handling. Supporting it upstream needs the same
  block-aligned-snapshot treatment mamba got — an architecture change, not config.
- Correctness note: even if the min() were relaxed, resuming from an external
  prefix would need the ring-buffer state snapshot at the resume boundary, which
  vLLM only keeps for the request's current tail.

**Watch for:** LMCache release notes mentioning circular-buffer /
`CircularBufferSpec` / Qwen-Next QSA support. When that lands, `LMCACHE=1` on the
current serve.sh should just work — the whole MP pipeline is already validated up
to the store decision.

## What survives in `scripts/serve.sh`

- `PREFIX_CACHE=1` (default): `--enable-prefix-caching --mamba-cache-mode align`.
  Upstream-supported for this hybrid model; measured **12–18 % GPU prefix-cache
  hits** on shared-prefix requests (the original serve.sh had prefix caching off).
  `PREFIX_CACHE=0` restores the old behavior.
- `LMCACHE=0` (default). `LMCACHE=1` starts/reuses the `lmcache-server` container
  (RAM L1 `LMC_L1_GB=5`, NVMe fs_native L2 `LMC_DISK_GB=150` at
  `~/run/lmcache-disk`, ZMQ `LMC_PORT=5557`, HTTP `LMC_HTTP_PORT=8090`,
  chunk `LMC_CHUNK=1600`, debug via `LMC_LOG=DEBUG`) and attaches vLLM via
  `LMCacheMPConnector` + `engine_driven`. Boots clean; inert until upstream fix.
- `DOCKER_EXTRA` hook: extra `docker run` args (used for the bind-mount
  instrumentation trick; generally useful).

## Useful diagnostics

```bash
# LMCache server health / stored-object counts / config
curl -s localhost:8090/status | python3 -m json.tool
curl -s localhost:8090/metrics | grep -v '^#' | grep -iE 'store|lookup|hit'

# vLLM side: per-step cache hit telemetry (log line 'loggers.py')
docker logs --since 5m qwen38-flash 2>&1 | grep 'Prefix cache hit rate'

# Inspect package source inside the image without booting anything
docker run --rm --entrypoint sh qwen38-flash-dgx -c 'grep -rn PATTERN /usr/local/lib/python3.12/dist-packages/lmcache/…'

# Instrumentation trick: extract a file, patch it, bind-mount it back
docker cp <ctr>:/usr/local/lib/python3.12/dist-packages/…/file.py /host/dir/
DOCKER_EXTRA='-v /host/dir/file.py:/usr/local/lib/…/file.py:ro' scripts/serve.sh
```

Gotchas hit along the way: gx10's login shell is **fish** (`$` in remote greps
needs care); `lmcache_mp_metadata.py` defines **no `logger`** (use `print(...,
flush=True)`); a first-request speedup can be JIT warmup (`jit_monitor` Triton
compile lines), not caching — verify with the hit-rate telemetry, not latency.

## Final state (2026-08-29)

- `qwen38-flash` serving on :8000 — `LMCACHE=0 PREFIX_CACHE=1 PORT=8000 MTP=3
  PREWARM=1 YARN=1 CTX=500000`.
- `lmcache-server` container removed (freed 5 GB of the unified pool);
  `~/run/lmcache-disk/` left in place (empty); `~/run/lmcache-debug/` holds the
  instrumented `lmcache_mp_metadata.py` (not mounted; kept for reference).

## Final validation (2026-08-29, LMCACHE=0 PREFIX_CACHE=1 MTP=3 PREWARM=1 YARN=1 CTX=500000)

- Boot: 12m39s (container start -> \"Application startup complete\", incl. PLE prewarm).
- smoke-test.sh localhost:8000: coherent, 1474 tok/s prefill (8k prompt), 42.4 tok/s decode.
- Prefix caching: ~6.8k-token shared prefix, steady-state TTFT 1.41s vs 2.05s cold
  (-31%); cumulative hit rate climbing as expected (13.3% after a handful of
  shared-prefix requests). First reuse can miss if sent immediately after the
  cold request; blocks land after a beat.
