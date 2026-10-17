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

Superseded (2026-10-17): the initial report described null-base pointer arithmetic
in graph sizing as a current upstream UBSan issue. Inspection of the caller,
ggml_graph_nbytes(), corrects that claim: current upstream starts its synthetic
pointer at 1024 and subtracts 1024 from the result, explicitly working around
the historical “applying non-zero offset X to null pointer” diagnostic. The old
fork instead changed the helper to integer arithmetic. This integration follows
upstream. UBSan has not been run on this pin; no current UBSan failure has been
demonstrated, and neither UBSan-clean status nor a present blocker is established.

Release status checked 2026-10-17: the standalone llama.cpp release
llama-mimir-4122b9a81 is still a draft with zero assets. The public
dfm-mimir-native-v0.1.5 release contains nine mimir-chat wrapper archives, not
the requested standalone llama-cli/server/bench/quantize distribution. Packaging
the integrated pin remains outstanding.

CUDA, Vulkan, Windows and Qualcomm runtime qualification must be repeated for
this new pin; evidence from the old pin does not automatically carry over.
The existing broad binary workflow still names the old pin and must be updated
before dispatch/publication. The native mimir-chat consumer also compiles successfully against the updated API
and generates “København.” using the v1.5 Q4_K_M model on Metal.
Apple application packages are not rebuilt or released by this integration.

Knowledge validation: uv run --no-project --with pyyaml python scripts/validate_okf.py wiki
completed with zero errors and 15 pre-existing missing-local-link warnings.
Use --no-project here to avoid installing the training project dependencies.

## Upstream published binary matrix (2026-10-17)

Verified actual assets of https://github.com/ggml-org/llama.cpp/releases/tag/b11706
against its release workflow and Snapdragon presets (both match the integrated
source files byte-for-byte). GitHub's releases/latest instead points to v0.7.0,
which currently contains only nightly-tag.txt; use the numbered build release.

- macOS: ARM64 CPU/Metal; x64 CPU (Metal explicitly disabled).
- Ubuntu x64: CPU, Vulkan, CUDA 12.8/13.4, ROCm 10.0, SYCL FP16/FP32,
  OpenVINO 2026.4.1.
- Ubuntu ARM64: CPU, Vulkan, CUDA 13.4; Ubuntu s390x: CPU.
- Linux ARM64 Snapdragon: OpenCL GPU and Hexagon NPU (release command overrides
  the preset's OpenCL OFF to ON).
- Windows x64: CPU, Vulkan, CUDA 12.4/13.4, ROCm 10.0, SYCL, OpenVINO 2026.4.1.
- Windows ARM64: CPU, Vulkan, CUDA 13.4, Adreno OpenCL. No Windows Hexagon
  package appears in this release.
- Android ARM64: generic CPU; Snapdragon OpenCL GPU plus Hexagon NPU.
- Additional assets: six CUDA runtime archives, web UI, Apple XCFramework
  built for macOS and iOS devices (not simulator).

These are published packages, not evidence that our PrefixLM/MixedLM execution
has been qualified on each accelerator. Reusing upstream packaging is preferable
to maintaining an independent matrix; hardware validation remains separate.

## Reused release workflow (2026-10-17)

Supersedes the earlier nine-target standalone packager: HRM-Text's
.github/workflows/llama-binaries.yml now derives build jobs, UI build and archive
assembly from the pinned upstream workflows. Regenerate with
`python3 native/binaries/sync_upstream_release.py`; `--check` detects drift.
Source checkout is pinned to the fork's full SHA, including in UI builds and
local composite actions. All upstream targets are retained except s390x, whose
upstream-specific runner is unavailable. Matrix fail-fast is disabled.

Manual dispatch accepts publish=false (default, Actions artifacts) or true
(draft upload followed by prerelease publication once every build passes).
Tags use mimir-<source-sha>; assets keep upstream's llama-<tag>-bin-* names,
Windows CPU/backend merge and separate CUDA runtimes. SOURCE.txt and SHA256SUMS
are added. The old mimir-chat publisher no longer handles standalone releases.
Standalone TLS follows upstream, not the Apple app's OS-only networking policy.
See native/binaries/LLAMA.md for operation and package limitations.

The generator passed YAML parsing, pinned-checkout/dependency validation and
deterministic regeneration checks. Cross-platform execution remains a CI gate;
successful packaging must not be confused with GPU/NPU inference qualification.

## Release warning fix (2026-10-17)

Initial matrix run 39003251048 failed Linux CPU x64/ARM64, macOS x64 and generic
Android builds on the same two signed/unsigned token-count comparisons in
decode_prefix_lm. Upstream enables LLAMA_FATAL_WARNINGS; earlier qualification
did not. The run was cancelled before publication to replace the source pin.

Fork commit 4f2e25bcf bounds the token count by INT32_MAX before converting it,
compares capacities without narrowing, and uses the checked signed count for
both loops. No attention semantics changed. The local strict Release build
passed for CLI, server, benchmark, quantizer and test-llama-archs. The Linux core
qualification workflow now also sets LLAMA_FATAL_WARNINGS=ON. Release jobs are
regenerated from the updated fork pin; no compiler warning suppression is used.
Both local CTest PrefixLM suites (HRMText and Llama) passed after this rebuild.

### Strict Linux qualification follow-up

Run 39003866775 exposed another warning in the PrefixLM test: GCC rejects a
copied range-loop pair under -Werror=range-loop-construct. Commit 1c979346d
uses a const reference; production runtime code is unchanged. Local strict
test compilation and both PrefixLM suites passed after the adjustment.

The ASan job also stopped during compilation on GCC 13 libstdc++ regex internals
with -Werror=maybe-uninitialized, before exercising our tests. ASan qualification
now uses Clang; the ordinary Release job retains GCC. Both retain fatal warnings,
and sanitizer flags and leak detection remain enabled. This is a compiler
selection change, not evidence that GCC ASan execution passed. Release matrix
39003864987 had no failed jobs at this check, but is superseded by a fresh run
against the updated test/source pin so distribution and qualification match.

### Shared-library ASan startup limitation

Run 39005079691: GCC Release passed. Clang ASan passed all five CTest suites
and all three ordinary architecture cases, then llama-cli --help aborted with
an ODR violation for httplib::detail::RegexMatcher RTTI. The duplicate globals
are in libllama-server-impl.so and libllama-common.so. Upstream's vendor target
builds cpp-httplib STATIC and links it into both shared libraries.

ASan qualification now uses the supported BUILD_SHARED_LIBS=OFF configuration
to give each executable a single copy. This does not disable ODR checks, fatal
warnings, leak detection or ASan. It does not establish shared-package ASan
startup cleanliness: that upstream linkage limitation remains unresolved.
Release qualification and published package linkage remain unchanged. No fork
runtime changes or release-matrix restart are needed for this CI-only adjustment.
