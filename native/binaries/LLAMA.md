# Mimir-compatible llama.cpp binaries

These are standalone tools from the pinned schneiderkamplab/llama.cpp fork,
including its HRMText, PrefixLM and MixedLM support. They contain no app or model
weights. The version in each filename identifies the llama.cpp source commit.

Included: llama-cli, llama-server, llama-bench, llama-quantize, shared libraries,
backend plugins, licenses, a source manifest and build-host device enumeration.

Download an official GGUF from danish-foundation-models/DFM-Mimir-v1.5-GGUF.
Keep all extracted files together. Use the model's embedded chat template:

    ./llama-cli -m /path/to/model.gguf --attention prefix-lm -cnv
    ./llama-server -m /path/to/model.gguf --attention prefix-lm --host 127.0.0.1 --port 8080

The server offers an OpenAI-compatible local API; it does not use OpenAI services.
Use --help for options. Benchmark and quantization tools are also included.
Windows executable names have an .exe suffix.

CPU support is included in GPU packages. Use -ngl 0 to request CPU inference.
CUDA packages require a compatible NVIDIA driver and CUDA 12.8 runtime/cuBLAS.
Vulkan packages require the platform Vulkan loader and a working GPU driver.
The macOS Metal package targets Apple Silicon. macOS requires version 14 or newer.
Linux packages target Ubuntu 22.04 or newer (Vulkan: Ubuntu 24.04 or newer).
Windows ARM64 CPU binaries can run on Snapdragon PCs; they do not accelerate
on Qualcomm's GPU or NPU. Qualcomm/OpenCL/Hexagon builds are not included.

CI verifies compilation, packaged help and device enumeration. GPU hardware and
model inference are not tested by hosted CI. See release notes for any additional
local qualification. macOS binaries are not Developer ID signed/notarized;
Windows binaries are not Authenticode signed. Windows may need the Visual C++
runtime. Checksums detect corruption, not publisher identity.
