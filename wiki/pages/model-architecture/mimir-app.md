---
type: Knowledge Collection
title: DFM Mimir app and portable backend selection
description: Separate Flutter client, shared native engine, local backend evidence and remaining platform qualification.
tags: [mimir, flutter, desktop, mobile, backends]
status: draft
last_updated: 2026-10-17
confidence: high
---
# DFM Mimir app and portable backends

The [Flutter client](../../../native/app/README.md) is separate from the
[SwiftUI app](mimir-apple-mvp.md), with bundle ID `dk.sdu.mimir` and a
separate local archive. Initial packaged targets are Apple Silicon Mac and arm64
iOS simulator. Android was initially only a runner scaffold; the Android
development integration below supersedes that status. Linux/Windows remain
runner scaffolds, not qualified builds.

Both clients now use `native/mimir/include/mimir/compaction.h`. The previous
Apple-local header was moved without algorithm changes, and the real SwiftUI
bridge regression suite passed. Flutter uses a C ABI with a native worker and
polled JSON events; teardown drains and joins away from the UI thread.

[Validation](../../../native/app/VALIDATION.md) records real generation,
cancellation/recovery, compaction streaming and storage tests, app builds, iOS
keyboard/send behavior and Mac exit checks. A diagnostic forced-semantics handle caused Mac AX-tree errors and was removed;
normal Flutter accessibility activation restored the live tree. A full
VoiceOver/TalkBack audit remains outstanding. Physical device, release
credentials and distribution work remain deferred.

The old CPU/Metal-only selector in `native/mimir/tools/model.h` is superseded by
registered backend/device selection (`auto`, `cpu`, backend name or exact device
name; `metal` aliases `MTL`). The llama.cpp PrefixLM kernels already use ggml's
backend abstraction; this extension does not alter the existing patch scopes.
Explicit unavailable selections fail. [Backend report](../../../native/mimir/BACKENDS.md)
records 73 functional checks each on Mac CPU/Metal/Accelerate/Vulkan. Existing
Linux CUDA evidence is retained, including its numerical acceptance and final
sanitizer-policy-runner caveats. Other backends are explicitly unqualified.

Verified build dependencies: Flutter 3.47.5, Dart 3.13.4, Xcode 16.3 and CocoaPods
1.17.0. No Xcode upgrade was needed. Framework slices must be copied inside each
pod root; CocoaPods did not discover the vendored XCFramework through a directory
symlink. Both runners intentionally build arm64. `build_native.py` reproduces
native packaging; model weights and generated binaries stay outside Git.


## Topic Pages

* [Portable App Packaging History](mimir-app-packaging-history.md) - Portable packaging, platform integration, feedback and compaction history.
* [App Model Catalog and Release History](mimir-app-model-catalog-history.md) - Model discovery, tokenizer corrections and 0.1.2 publication history.
* [App UI and Generation Defaults](mimir-app-ui-defaults.md) - Sampling defaults, branding, prompt identity and text rendering decisions.
* [Native CLI Distribution](mimir-native-cli-distribution.md) - Native CLI packaging, standalone llama.cpp releases and platform qualification.

## Historical Section Links

Original headings remain here for incoming links; details live in the topic pages.

## Startup chat creation — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#startup-chat-creation--2026-09-20).

## Android development integration — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#android-development-integration--2026-09-20).

## MixedLM and Xcode update — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#mixedlm-and-xcode-update--2026-09-20).

## Proposed cross-platform release plan — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#proposed-cross-platform-release-plan--2026-09-20).

## Portable package implementation — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#portable-package-implementation--2026-09-20).

### Bundled weights — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#bundled-weights--2026-09-20).

### Flutter macOS DMG — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#flutter-macos-dmg--2026-09-20).

### Android Vulkan enabled — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#android-vulkan-enabled--2026-09-20).

### Distribution filenames — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#distribution-filenames--2026-09-20).

### API availability — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#api-availability--2026-09-20).

### Desktop API implementation — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#desktop-api-implementation--2026-09-20).

### Display branding — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#display-branding--2026-09-20).

### Feedback collection decision — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#feedback-collection-decision--2026-09-20).

### Feedback D1 provisioning — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#feedback-d1-provisioning--2026-09-20).

### Feedback implementation — 2026-09-20

See [Portable App Packaging History](mimir-app-packaging-history.md#feedback-implementation--2026-09-20).

### Android startup recovery — 2026-09-21

See [Portable App Packaging History](mimir-app-packaging-history.md#android-startup-recovery--2026-09-21).

### Android 0.1.1 publication — 2026-09-21

See [Portable App Packaging History](mimir-app-packaging-history.md#android-011-publication--2026-09-21).

### Bounded prompt and history compaction — 2026-09-21

See [Portable App Packaging History](mimir-app-packaging-history.md#bounded-prompt-and-history-compaction--2026-09-21).

### Complete 0.1.1 preview refresh — 2026-09-21

See [Portable App Packaging History](mimir-app-packaging-history.md#complete-011-preview-refresh--2026-09-21).

## 2026-09-21: optional model library and public checkpoint inventory

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#2026-09-21-optional-model-library-and-public-checkpoint-inventory).

### Official GGUF discovery extension, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#official-gguf-discovery-extension-2026-09-21).

### Training tokenizer reference correction, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#training-tokenizer-reference-correction-2026-09-21).

### Corrected 4/8/16-bit artifacts prepared, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#corrected-4816-bit-artifacts-prepared-2026-09-21).

### Official GGUF publication completed, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#official-gguf-publication-completed-2026-09-21).

### Official-only model catalog policy, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#official-only-model-catalog-policy-2026-09-21).

### Explicit third-party repository selection, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#explicit-third-party-repository-selection-2026-09-21).

### 0.1.2 package preparation, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#012-package-preparation-2026-09-21).

### 0.1.2 publication completed, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#012-publication-completed-2026-09-21).

### Separate release history, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#separate-release-history-2026-09-21).

### Self-contained release descriptions, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#self-contained-release-descriptions-2026-09-21).

### Release-description policy for all versions, 2026-09-21

See [App Model Catalog and Release History](mimir-app-model-catalog-history.md#release-description-policy-for-all-versions-2026-09-21).

## MixedLM default and reply sampling — 2026-09-22

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#mixedlm-default-and-reply-sampling--2026-09-22).

## Basic Markdown rendering — 2026-09-22

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#basic-markdown-rendering--2026-09-22).

## Stronger default repetition penalty — 2026-09-22

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#stronger-default-repetition-penalty--2026-09-22).

## 0.1.3 packaging — 2026-09-22

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#013-packaging--2026-09-22).

### 0.1.3 publication completed

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#013-publication-completed).

## 0.1.4 preparation: UCloud training credit — 2026-09-22

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#014-preparation-ucloud-training-credit--2026-09-22).

## Concise default model identity — 2026-09-23

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#concise-default-model-identity--2026-09-23).

## Search and saved prompt correction — 2026-09-23

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#search-and-saved-prompt-correction--2026-09-23).

## Default training languages — 2026-09-23

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#default-training-languages--2026-09-23).

## Appearance and settings

See [app appearance and settings](mimir-app-appearance.md).

## Native binaries

See [native binary distribution](mimir-native-binaries.md) and the
[upstream integration](prefixlm-upstream-integration.md) for standalone tools and qualification.

## Earlier UI and native distribution history

See [App UI and Generation Defaults](mimir-app-ui-defaults.md#app-text-scaling-2026-09-24).

## Native CLI distribution — 16 October 2026

See [Native CLI Distribution](mimir-native-cli-distribution.md#native-cli-distribution--16-october-2026).

### 2026-10-17: standalone llama.cpp deliverable correction

See [Native CLI Distribution](mimir-native-cli-distribution.md#2026-10-17-standalone-llamacpp-deliverable-correction).

### 2026-10-17: upstream refresh and Qualcomm release gate

See [Native CLI Distribution](mimir-native-cli-distribution.md#2026-10-17-upstream-refresh-and-qualcomm-release-gate).
