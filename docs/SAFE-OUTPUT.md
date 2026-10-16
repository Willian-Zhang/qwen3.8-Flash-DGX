# Qwen parser and Responses output integrity (patch 19)

Patch 19 ports [Chaos2522's review2 source bundle](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6191209070)
from vLLM v0.30 to the recipe's v0.31.0 base. It applies after patches 12 and 13.
Credit for the original fixes and regression fixtures belongs to Chaos2522.
Archive provenance and the eight source-file hashes are in
[`SOURCE.json`](../tests/issue31_safe_output/SOURCE.json) and
[`PORT-MANIFEST.json`](../tests/issue31_safe_output/PORT-MANIFEST.json).

## Behavior

- Final Qwen parsing replays the original token IDs. Ordinary BPE text spelling
  a protocol marker stays ordinary text, matching streaming behavior; actual
  atomic control tokens retain their protocol meaning.
- Responses generations ending in STOP with no visible answer or function call
  fail explicitly. This includes whitespace-only output and hidden reasoning
  without a visible answer. Streaming emits `response.failed` with an error;
  nonstreaming returns an HTTP 500 error.
- LENGTH keeps incomplete status and emits `response.incomplete`, rather than
  `response.completed` with an incomplete payload. The declared event schema
  and background replay recognize all three terminal outcomes.
- The bundle also hardens Qwen tool-name validation, known-namespace handling,
  reasoning/tool transitions and malformed-opener recovery. Unknown names are
  not mapped arbitrarily to an offered tool.

The v0.31 port scopes ID-aware final replay to the Qwen3 reasoning engine,
preserving other parser configurations. Chat Completions retains incremental
argument streaming. Responses buffers converted XML arguments until call
closure so a truncated XML call does not emit unfinished JSON arguments.
Clients consuming Responses argument deltas will therefore receive arguments
later than before.

The patch changes neither weights nor sampling, chat templates, KV precision,
MTP or serving limits. It cannot recover content the model never generated,
prevent premature EOS, or decide whether a nonempty final sentence means a task
is actually finished. The contributor's optional native sampler guard is not
included.

## GX10 validation, 2026-10-15–16

The comparison used vLLM v0.31.0, the NVIDIA hybrid checkpoint, MTP=2,
YaRN 500k, prefix caching, PLE prewarm, CPUSET `5-9,15-19`, 16 sequences,
BF16 recurrent state and FP8 KV with a fixed 8 GiB pool. Both arms used the
same weights and launch configuration. Ollama and LiteLLM remained resident;
external gateway traffic was not blocked during the baseline.

### Targeted checks

- All **67 regression methods** and **21 literal-byte fixtures** pass against
  installed candidate modules, without runtime source overlays. The only fixture
  adaptation initializes v0.31's `enable_per_request_metrics=False`; assertions
  are retained.
- The official parser-engine suite plus the existing patch-12/13 tests gives
  **4,781/4,782 passing in both arms**. The existing failure is
  `TestTruncatedToolOpenerStreamParity::test_content_around_unpromoted_tool_block_stays_ordered`:
  upstream expects malformed marker text to disappear, while this recipe's
  literal-marker guard preserves it. No newly failing cases were introduced.
- Real inference reproduces the baseline's wrong LENGTH terminal event and its
  successful empty STOP. The candidate emits `response.incomplete` for visible
  and hidden-reasoning LENGTH, and fails empty STOP in streaming and nonstreaming
  modes. Empty STOP was induced with a client stop string; this tests the serving
  contract, not the frequency or cause of natural premature EOS.
- Normal visible answers, tool calls and tool-result round trips pass in both
  modes. Separate model-driven literal-copy probes remain **4/8 in both arms**;
  these do not isolate model emission from parsing and are not evidence that
  all literal-copy problems are fixed.
- The actual Codex CLI `0.162.0-alpha.17.2`, replaying synthetic streams exported
  by the installed producer, handles all **4/4** candidate outcomes correctly
  versus **1/4** baseline outcomes. LENGTH and empty STOP become visible failed
  turns; normal STOP succeeds. This uses synthetic engine frames, not model
  inference or a coding task; each case makes one request and executes no tools.

See the [CPU/client reproduction instructions](../tests/issue31_safe_output/README.md).

### Performance and general regression check

| Measurement | Baseline | Patch 19 |
|---|---:|---:|
| Single-request decode, median of six | 47.95 tok/s | 47.08 tok/s |
| Aggregate throughput, 8 clients | 220.97 tok/s | 226.68 tok/s |
| Aggregate throughput, 16 clients | 362.28 tok/s | 364.20 tok/s |
| Aggregate throughput, 32 clients | 321.95 tok/s | 327.36 tok/s |
| Uncached prefill, ~8.5k tokens, time to first token | 2.99 s | 3.97 s |
| Uncached prefill, ~34k tokens, time to first token | 10.99 s | 13.03 s |
| Uncached prefill, ~101k tokens, time to first token | 32.88 s | 35.09 s |
| 55-scenario tournament, one pass | 87.3% | 90.8% |

Speed probes explicitly disabled thinking. Parallel values are medians of two
sweeps. Prefill used fresh prompts with zero cached tokens, one measurement per
size per arm. Prefill was slower in this comparison; a repeated controlled
comparison is needed to attribute the difference to the patch. Decode and
aggregate throughput stayed close. No speedup is claimed for these fixes.

The unchanged tournament used temperature 0.2, four parallel scenarios and a
32,000-token response cap. Both passes completed without technical errors.
Token-limit finishes were seven in the baseline and five in the candidate.
The tournament serves as a general regression check; its score does not measure
the bugs corrected here. One stochastic pass per arm does not establish equal
quality or a model improvement.
