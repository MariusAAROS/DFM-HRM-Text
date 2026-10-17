# DFM Mimir native command-line binaries

These packages contain `mimir-chat` and the matching libraries from our pinned
PrefixLM/MixedLM llama.cpp fork. They are not Flutter GUI installers. Model
weights are downloaded separately from
https://huggingface.co/danish-foundation-models/DFM-Mimir-v1.5-GGUF .
Use the corrected official Q4_K_M model for a compact starting point.

Extract the whole archive, keeping all libraries next to the executable:

```
./mimir-chat --list-devices
./mimir-chat --model /path/to/model.gguf --device cpu --ctx 2048 --batch 2048
./mimir-chat --model /path/to/model.gguf --device auto --ctx 2048 --batch 2048
```

Windows uses `mimir-chat.exe`. This client uses the model's chat template and
PrefixLM metadata; it is a greedy terminal client, not an HTTP server. `/quit`
exits and `/reset` starts a new conversation. The included engine contains
MixedLM support, but this CLI does not expose a MixedLM switch.

Every package includes CPU support. Accelerator backends are dynamically loaded;
CPU-only packages avoid GPU driver dependencies. Use `--device cpu` explicitly
when diagnosing driver problems. CUDA needs a compatible NVIDIA driver and CUDA 12.8 runtime/cuBLAS libraries; Vulkan
needs a Vulkan-capable driver; OpenCL needs the vendor's OpenCL runtime. A
successful build is not a guarantee of model correctness on untested hardware.
See manifest.json for provenance and validation scope, and devices.txt for the
build runner's device enumeration. Hosted CI normally has no usable GPU.

macOS binaries are not Developer ID signed/notarized. Windows binaries are not
Authenticode signed. Linux builds target the runner's glibc baseline (Ubuntu
22.04 x64; Ubuntu 24.04 ARM64). Windows may need the Microsoft Visual C++ runtime.

Build locally with Python 3 and CMake:
`python native/binaries/package.py --backend cpu`.
Vulkan, CUDA and OpenCL builds additionally need their SDK/toolchain installed.
