---
type: Runbook
title: PrefixLM and MixedLM upstream integration
description: Scoped integration onto October 2026 upstream llama.cpp, API adaptation, qualification evidence and release gates.
tags: [mimir, llama-cpp, prefixlm, mixedlm, release]
status: draft
last_updated: 2026-10-17
confidence: high
---

# PrefixLM and MixedLM upstream integration

The standalone binary release is paused until this integration is qualified.
This supersedes the preliminary merge assessment in [the binary runbook](mimir-native-binaries.md):
the final fetched upstream master is f74e2fa6a31cc33ab4ee4ae516641cfe62df3924,
not the earlier ea0c4c2c8 observation. A remote master lookup confirmed this
pin on 2026-10-17. There are 688 upstream-only commits and eight old fork-only
commits relative to codex/mixedlm.

## Scope and history

The integration is on llama.cpp branch codex/prefixlm-upstream-20261017,
based directly on upstream, with two scoped commits:

- 3eed91acf: training-faithful Mimir tokenizer/converter and Unicode template support.
- a800a5ff3: PrefixLM/MixedLM, lifecycle rules, exact-prefix cache, extended-batch API adaptation and regression tests.

The consuming HRM-Text branch is codex/llama-upstream-integration.
Main and the old codex/mixedlm branch are unchanged. No history was rewritten
and no upstream PR was submitted.

Per the user's scope instruction, generic ggml pointer-arithmetic changes,
Vulkan SDK/Android fixes and the separate sampler/server generation-persistence
extension were not ported. Those remain recoverable from the old branch.
PrefixLM KV/context snapshots and boundary logits remain in the attention patch;
sampler serialization, retained generations across server restarts and parser
continuation follow upstream instead. These should not be advertised as features
of the new standalone build.

## Upstream attention support

The fetched upstream public enum supports only causal and non-causal attention.
HRMText's prefix metadata is retained for round trips, with an explicit comment
that PrefixLM prefill is not implemented. No MixedLM implementation is present.
Thus neither attention implementation can currently be replaced with upstream
functionality.

Upstream has introduced llama_batch_ext and llama_process, used by its CLI,
server and examples. The integrated attention implementation validates extended
batches directly. Added entry points are llama_process_prefix,
llama_process_prefix_mixed, and llama_process_mixed_lm. Legacy prefix APIs
prepare extended batches with per-sequence inferred positions. Extended callers
set positions explicitly. Ordinary encode/decode behaviour follows upstream;
both encode APIs reject PrefixLM and both decode APIs enforce continuation rules.

Unified PrefixLM retains sparse sequence IDs with a bounded number of live
sequences. Tests of ordinary causal sparse IDs now allocate upstream's required
ID range explicitly. No causal-mode batch namespace was expanded.

## Local qualification

Build: Apple M2 Max, Release, CPU/Accelerate/Metal, OpenSSL disabled.
Built CLI, server, bench, quantizer, simple/simple-chat examples and selected tests.

- HRMText PrefixLM/MixedLM suite: **2,144 checks passed**.
- Llama PrefixLM/MixedLM suite: **2,144 checks passed**.
- Both suites exercise CPU and Metal, flash attention on/off, legacy and extended
  calls, independent block-attention oracle, forks, shared capacity, state restore,
  mixed phases, adapters and invalid input.
- Upstream ordinary Llama/HRMText architecture checks: **12 passed**. BERT entries
  were skipped by that test harness; this is not an encoder qualification.
- CTest sampling, batch allocation and chat tests passed. The model-dependent
  backend-sampler CTest exit alone is not claimed as model-based qualification.
- Existing independent HF tiny-model reference: **31 steps / 130 output rows**.
  CPU max absolute error 7.152557373046875e-7; Metal without flash
  8.344650268554688e-7; both pass the original 1e-4 tolerance.
  Metal flash max error 1.0834634304046631e-4; two causal-control steps fail that
  strict threshold. All 130 top-token choices agree on each backend. No tolerance
  was relaxed. The prior recorded Metal-flash run also failed the same strict
  threshold (maximum 7.294714450836182e-4).
- Real v1.5 CLI chat using embedded template: Q4_K_M passed on CPU and Metal;
  Q8_0 and BF16 passed on Metal. Each answered “København”.
  Upstream CLI no longer accepts -cnv; chat is its native interface.
- Server OpenAI-compatible chat: cold/warm exact-prefix requests, two concurrent
  slots and multi-turn follow-up passed. Warm cache reported prompt_n=0,
  cache_n=29, with the same response as cold generation.

Local machine-readable results: logs/upstream-integration/.
Build: logs/llama-upstream-build/. Diagnostic logs: /tmp/llama-upstream-*
and /tmp/prefix-{hrm,llama}-upstream.log.

## Linux gate and remaining work

Dispatch .github/workflows/mimir-runtime.yml on the integration branch with
core_only=true (the default on this branch). Pull requests also use this core
qualification path. This runs standalone Linux CPU Release and AddressSanitizer
builds, both PrefixLM suites, ordinary architecture checks and helper tests.
It does not run the historical app/persistence qualification workflow.
Initial CI run: https://github.com/schneiderkamplab/HRM-Text/actions/runs/38997326649.
Linux Release and AddressSanitizer both passed: five CTest suites per build,
1,072 PrefixLM/MixedLM checks for each of HRMText and Llama per build, plus
three ordinary CPU architecture cases per build. ASan used halt_on_error=1
and detect_leaks=1. CI completed successfully on 2026-10-17; downloaded test
logs are in logs/upstream-integration/linux-release and linux-asan.

UBSan is a separate unresolved upstream issue: ggml/src/ggml.c still performs
null-base pointer arithmetic in incr_ptr_aligned during graph sizing. The old
fork fixed it independently; this integration deliberately follows upstream for
that unrelated issue. Do not describe this build as UBSan-clean.

CUDA, Vulkan, Windows and Qualcomm runtime qualification must be repeated for
this new pin; evidence from the old pin does not automatically carry over.
The existing broad binary workflow still names the old pin and must be updated
before dispatch/publication. The native mimir-chat consumer also compiles successfully against the updated API
and generates “København.” using the v1.5 Q4_K_M model on Metal.
Apple application packages are not rebuilt or released by this integration.

Knowledge validation: uv run --no-project --with pyyaml python scripts/validate_okf.py wiki
completed with zero errors and 15 pre-existing missing-local-link warnings.
Use --no-project here to avoid installing the training project dependencies.
