# Mimir-compatible llama.cpp binaries

Standalone tools from the pinned schneiderkamplab/llama.cpp fork, including
HRMText, PrefixLM and opt-in MixedLM. No app installer or model weights.

## Release workflow

`.github/workflows/llama-binaries.yml` reuses the pinned upstream release jobs,
UI build, Windows CPU/backend archive merging and platform packaging. Regenerate
it after updating the submodule:

```sh
python3 native/binaries/sync_upstream_release.py
python3 native/binaries/sync_upstream_release.py --check
```

The generator keeps upstream build commands and dependencies. Local adaptations:
all source checkouts use the fork's full commit SHA; releases are manual and use
`mimir-<short-source-sha>` tags in HRM-Text; the unavailable upstream-specific
s390x runner is omitted. Failed matrix entries do not cancel other builds.
No upstream deployment keys, Hugging Face publishing or master-branch changes
are needed. The separate native mimir-chat publisher does not handle these assets.

```sh
gh workflow run llama-binaries.yml --ref codex/llama-upstream-integration -f publish=true
```

With `publish=false` (default), completed packages are retained as the
`release-packages` Actions artifact. With `publish=true`, all build jobs must pass
before publication. Assets are uploaded to a draft, then published as a prerelease;
a retry can complete a draft but refuses to replace a published release.
SOURCE.txt identifies source and packaging revisions; SHA256SUMS covers archives.

Packages follow upstream names, e.g.
`llama-mimir-<sha>-bin-macos-arm64.tar.gz`,
`llama-mimir-<sha>-bin-win-vulkan-arm64.zip`.
Windows CUDA runtime archives retain upstream's unversioned cudart filenames.

## Matrix

- macOS ARM64 CPU/Metal and x64 CPU; Apple XCFramework for macOS/iOS devices.
- Ubuntu x64 CPU, Vulkan, CUDA 12.8/13.4, ROCm, SYCL FP16/FP32, OpenVINO.
- Ubuntu ARM64 CPU, Vulkan, CUDA 13.4.
- Linux ARM64 Snapdragon OpenCL/Hexagon.
- Windows x64 CPU, Vulkan, CUDA 12.4/13.4, ROCm, SYCL, OpenVINO.
- Windows ARM64 CPU, Vulkan, CUDA 13.4, Adreno OpenCL.
- Android ARM64 CPU and separate Snapdragon OpenCL/Hexagon.
- Separate CUDA runtime archives and server web UI.

SDK versions and OS baselines follow the pinned upstream workflow. Snapdragon
packages require compatible hardware/OS support; ARM64 alone is insufficient.
A successful build does not establish PrefixLM/MixedLM correctness on that GPU/NPU.
The integration report records hardware validation separately.

## Usage

Download a model from danish-foundation-models/DFM-Mimir-v1.5-GGUF, keep archive
contents together and use its embedded chat template:

```sh
./llama-cli -m /path/to/model.gguf --attention prefix-lm
./llama-server -m /path/to/model.gguf --attention prefix-lm --host 127.0.0.1 --port 8080
```

Windows executables have an `.exe` suffix. The server offers an OpenAI-compatible
local API; it does not use OpenAI services. Standalone packages follow upstream
TLS defaults, separate from the Apple app's OS-only networking requirements.
GPU packages include CPU support. Use `-ngl 0` for CPU inference. CUDA packages
need compatible drivers and matching runtime libraries (separate assets).
These packages are not Developer ID notarized or Authenticode signed.

`package.py --product llama` remains a local minimal-toolset build helper; it is
not the release packaging path and has different TLS and platform defaults.
