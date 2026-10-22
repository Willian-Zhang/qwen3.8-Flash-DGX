# Qwen parser and Responses output integrity (patches 19 and 21)

Patch 19 ports [Chaos2522's review2 source bundle](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6191209070)
from vLLM v0.30 to the then-current v0.31.0 base. It applies after patches 12 and 13.
Credit for the original fixes and regression fixtures belongs to Chaos2522.
Archive provenance and the eight patch-19 source-file hashes are in
[`SOURCE.json`](../tests/issue31_safe_output/SOURCE.json) and
[`PORT-MANIFEST.json`](../tests/issue31_safe_output/PORT-MANIFEST.json).
The patch-21 follow-up and historical v0.31 hashes are described below.
For the current v0.32 rebase, installed-source manifest and validation, see
[VLLM-032.md](VLLM-032.md).

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
- Responses also hardens Qwen tool-name validation. The parser handles known namespaces,
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

## Chat Completions compatibility follow-up (issue #56)

Patch 19 initially enforced the offered-name list for both APIs. With synthetic
engine frames passed through the installed Chat Completions producer, an
unoffered call became an empty turn ending in `stop`; an unoffered call followed
by a valid call could leave the first delivered tool at index 1. Before patch 19,
Chat Completions returned the unoffered name and used `tool_calls`, allowing the
client to handle an unavailable tool.

The follow-up restores that Chat contract for syntactically valid tool names.
Known offered names and namespace aliases resolve first; an unknown name is
passed through unchanged, never remapped to an offered function. Clients must
validate tool names before executing calls. Responses retains strict validation
and its explicit failure events. The choice is per parser/request, leaving the
shared cached configuration untouched. Rejection diagnostics include a bounded,
escaped tool name without logging arguments.

The `GenerationError` checks cited in the issue belong to Responses serving,
not Chat Completions. The installed Chat generator completed the synthetic
reproductions with a normal `[DONE]`; a cross-request server abort was not
reproduced. The compatibility tests cover the empty-turn change, call indices,
full/streaming parity, namespace resolution, preserved reasoning examples and
independent concurrent streams. They exercise actual installed serving code
with synthetic engine frames, not Open WebUI or long-context model inference.
The follow-up has 10 Chat compatibility regression methods; patch 20 has four
separate MTP configuration checks. These supplement the original patch-19 suite
below; its GPU measurements describe the original port, not this follow-up.
On the isolated GX10 CPU image, all 10 + 4 checks pass, the original 67 methods
and 21 byte fixtures still pass, and the official parser-engine result remains
4,781/4,782 with exactly the pre-existing failure documented below. The MTP test
reproduces the inherited-aux configuration error on the old image and clears it
with the fix; it does not exercise GPU model startup.

## Schema-authoritative tool arguments (patch 21)

Patch 21 ports the bounded argument normalization from
[Chaos2522's schema2 bundle](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6291191080)
to the existing v0.31 implementation. The original `argument_validation.py` helper
is copied unchanged; adapters and serving checks are integrated with patch 19 and
the issue #56 Chat compatibility contract. This is not a wholesale replacement
with the bundle's ten runtime files.

The uniquely offered tool schema is authoritative. An original payload that
already validates is preserved exactly. Otherwise, only a sole `arguments`
wrapper may be decoded once, and only if its inner object validates in full.
Missing or ambiguous schemas do not authorize repair. Legitimate `arguments`
properties, sibling properties, required fields, nested types and local schema
references are covered. Duplicate JSON keys, nonfinite values and unsupported
schemas cannot authorize repair; remote schema references are never fetched.

Responses withholds known-schema calls until explicit closure and validation,
then either emits the valid call or reports a schema-specific error before an
executable function-call item. LENGTH without a closer stays incomplete. Chat
retains its invalid-argument client recovery and unknown-name passthrough. Normal
Chat arguments still stream incrementally; possible wrappers wait until closure.
The schema catalog is per parser/request and cached without reusing a prior
request's schema. Implicit reasoning-to-tool transitions use the same checks.

GX10 installed-code CPU checks on 2026-10-20 give **22/23 author schema methods,
11/11 producer integration methods, 10/10 Chat compatibility methods, 67/67 prior
regression methods and 21/21 byte-fidelity cases**. The author's assertions are
retained with a v0.31 fixture adapter. Its namespace test expects dotted aliases
already rejected by the reference; the canonical `namespace__function` form
works in both arms and is repaired by the candidate. The official parser-engine
suite remains **4,781/4,782**, with the same pre-existing literal-marker failure
documented below. No new failures were introduced in this matrix.

These checks use synthetic engine frames through installed parser and serving
code, without runtime source overlays, model inference or tool execution. The
author's full 63-test matrix, live Codex execution, GPU performance and long-context
generations have not been validated for this port. The older GPU/tournament
measurements below describe patch 19, not patch 21. No EOS or model-quality
improvement is claimed.

See the [runner and known failures](../tests/issue31_schema_arguments/README.md),
[provenance](../tests/issue31_schema_arguments/SOURCE.json),
[v0.31 installed hashes](../tests/issue31_schema_arguments/PORT-MANIFEST.json)
and [recorded validation](../tests/issue31_schema_arguments/VALIDATION.json).

## GX10 validation, 2026-10-15–16 (patch 19)

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
