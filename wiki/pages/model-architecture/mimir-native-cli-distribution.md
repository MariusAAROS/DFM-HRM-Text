---
type: Technical Reference
title: Native CLI Distribution
description: Native CLI packaging, standalone llama.cpp releases and platform qualification.
tags: [mimir, app, releases]
status: draft
last_updated: 2026-10-17
confidence: high
---
# Native CLI Distribution

Part of [DFM Mimir app and portable backends](mimir-app.md).

## Native CLI distribution — 16 October 2026

`native/binaries/package.py` and `.github/workflows/mimir-binaries.yml` add
standalone mimir-chat archives with dynamic backend libraries, CPU fallback,
licenses, checksums and source manifests. These are not Flutter installers and
do not bundle weights. The CLI uses model chat templates and PrefixLM; the
engine contains MixedLM but the CLI has no mode switch. The initial matrix is
macOS ARM64 Metal/x64 CPU, Linux and Windows ARM64/x64 CPU, Linux ARM64/x64
Vulkan, and Linux x64 CUDA. Packaging validates policy tests and executable
startup/device enumeration; these checks do not qualify inference on GPU
hardware absent from hosted runners. CUDA packages require external CUDA runtime
and cuBLAS libraries. Qualcomm/Hexagon builds remain outside this initial matrix.

The first CI pass exposed missing Vulkan SPIR-V headers/Ubuntu 22.04 glslc,
incorrect CUDA apt package names, and the Windows ARM64 requirement for ClangCL.
The corrected matrix uses Ubuntu 24.04 for Vulkan and explicit CUDA/cuBLAS
packages. An archive audit also found absolute build-directory rpaths; packaging
now uses `@loader_path` on macOS and `$ORIGIN` on Linux. The extracted macOS ARM64
archive was tested from a temporary directory with v1.5 Q4_K_M on CPU and Metal;
both generated a correct Danish capital answer using the embedded chat template.
This is a smoke test, not broad numerical qualification. The native CLI preview
is published under GitHub tag `dfm-mimir-native-v0.1.5`; additional CI artifacts
must pass their checks before being attached.

As of 17 October, eight archives are published: macOS ARM64 Metal/x64 CPU,
Linux ARM64/x64 CPU and Vulkan, Windows ARM64/x64 CPU. CI run `38972076345`
has passed all jobs except Linux x64 CUDA, which is still compiling at this
observation. Downloaded archive checksums, license files and manifests were
verified. `mimir-binaries-publish.yml` attaches checksum-verified artifacts to the
existing native preview only after a successful, manually dispatched main-branch
matrix run from this repository. It never executes downloaded artifact contents.
CUDA publication remains contingent on CI success, not implied by this note.

### 2026-10-17: standalone llama.cpp deliverable correction

The requested broad-platform binaries are llama.cpp tools, not the native
mimir-chat wrapper. The earlier native preview remains a separate deliverable.
The shared packager now accepts --product llama and builds directly from the
pinned llama.cpp checkout, packaging llama-cli, llama-server, llama-bench and
llama-quantize. The separate llama-binaries workflow covers macOS ARM64 Metal
and x64 CPU, Linux ARM64/x64 CPU and Vulkan, Linux x64 CUDA, and Windows
ARM64/x64 CPU. Artifact versions identify the llama.cpp commit rather than the
application version. Qualcomm GPU/NPU acceleration remains unqualified and is
not represented by the Windows ARM64 CPU package. Initial build qualification
is in progress; hosted GPU builds alone do not demonstrate GPU inference.

### 2026-10-17: upstream refresh and Qualcomm release gate

Publication of the standalone llama.cpp preview was paused following the user's
upstream-refresh question; the empty release was returned to draft and build
run 38992809042 was cancelled. No new llama.cpp archives were published.
A read-only merge-tree check against upstream ea0c4c2c8 found 18 conflicted files.
The pinned fork has eight unique commits; upstream has 687 since the common
ancestor c9a5eeeb34ab8f794ea7510ca52d25da13728a5b. Conflicts include context,
KV cache, sampling, server state, tokenizer, Vulkan and tests. No actual merge
was applied. Refresh on an integration branch and rerun PrefixLM/MixedLM,
tokenization, lifecycle/server persistence and ordinary causal regressions
before choosing a new binary release pin.

Qualcomm code is already present in the pinned fork. Adreno OpenCL packaging
needs the SDK/loader and real-device qualification. Hexagon additionally needs
its SDK/toolchain, architecture-specific HTP libraries and compatible device
drivers/FastRPC runtime. Windows Hexagon requires signed HTP catalog files;
the upstream developer recipe uses test-signing (potentially disabling Secure
Boot), so a public plug-and-play NPU release needs a separate signing solution.
Existing Windows ARM64 hosted CI only establishes CPU functionality. Test real
Mimir Q4_K_M, Q8_0 and 16-bit models with PrefixLM/MixedLM, explicit masks,
fallback and memory limits before declaring either backend supported.

References: upstream docs/backend/OPENCL.md and
docs/backend/snapdragon/{README,windows,linux}.md, inspected locally and on
https://github.com/ggml-org/llama.cpp on 2026-10-17.
