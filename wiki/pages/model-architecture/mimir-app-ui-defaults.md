---
type: Technical Reference
title: App UI and Generation Defaults
description: Sampling defaults, branding, prompt identity and text rendering decisions.
tags: [mimir, app, releases]
status: draft
last_updated: 2026-10-17
confidence: high
---
# App UI and Generation Defaults

Part of [DFM Mimir app and portable backends](mimir-app.md).

## MixedLM default and reply sampling — 2026-09-22

**Supersedes the September 20 off-by-default Flutter setting:** MixedLM is now
on for new settings and archives without a saved mode; explicit saved choices
remain unchanged. Native library/API defaults remain exact PrefixLM.

Flutter settings expose temperature 0–2 (default 0) and repetition penalty 1–2
(default 1, disabled). They persist, validate on restore, and are forwarded per
reply without model reload. The native sampler applies the repetition penalty
before temperature/selection, using the last 64 generated tokens of the current
reply and resetting each reply. Prompts and prior turns are not seeded into the
penalty history. Compaction continues using deterministic defaults; API request
sampling remains independent of UI preferences.

Validation: Flutter analysis and all 48 tests pass. A fresh CPU build of
`MimirRuntime` and `mimir-text-tests` passed 78 text checks with the bundled
corrected Q4_K_M model, including seeded temperature/penalty replay and invalid
penalty rejection without history mutation. An asynchronous C-ABI smoke check
confirmed non-default sampling, subsequent MixedLM prefix reuse, and invalid
penalty rejection. Logs are `logs/sampling-native-tests.log` and
`logs/sampling-ffi-tests.log`. Packaged applications have not been rebuilt for
this change.

## Basic Markdown rendering — 2026-09-22

Flutter assistant replies, streamed replies, visible compaction summaries and
shortened-prompt previews use a shared selectable Markdown renderer. User text
stays literal, and transcript persistence/sharing remains unchanged. The pinned
`flutter_markdown_plus` 1.0.12 dependency supplies rendering; see its
[widget documentation](https://pub.dev/packages/flutter_markdown_plus).
Image builders always return text placeholders (no remote fetch or local-file
read). Links show an inspect/copy dialog without launching anything.

Validation: Flutter analysis and all 51 tests pass. New widget coverage checks
basic blocks at 320-pixel width, literal user messages, incomplete streaming
markup, image placeholders and link inspection. Release packages are not yet
rebuilt.

## Stronger default repetition penalty — 2026-09-22

**Supersedes the 1.0 UI default above:** the Flutter default is now 1.1. Archives
without the sampling-defaults marker upgrade a stored 1.0 once; other custom
values remain unchanged. Saving records the marker so a subsequent explicit
choice of 1.0 remains disabled after reopening. Native/API defaults and the
64-generated-token window are unchanged. This is a tuning choice prompted by
observed repetitions, not evidence of eliminating repetition in every reply.

## 0.1.3 packaging — 2026-09-22

Release source `2137f438786b79aeca16b70576d7c753d3ab0389` is version 0.1.3+5.
All four packages include the same official corrected Q4_K_M as 0.1.2. Features:
MixedLM default for new settings, persisted temperature and repetition penalty
(default 1.1 with one-time old-default upgrade), and offline Markdown rendering.
[Self-contained release notes](../../../native/app/releases/0.1.3.md) include API,
headless use, model selection, privacy, platform limits and package checksums.

[CI 35707335507](https://github.com/schneiderkamplab/HRM-Text/actions/runs/35707335507)
passed on Linux and Windows: 52 tests, analysis, native fallback/compaction and
packaged CPU backend probes. macOS and Android ARM64 builds passed locally.
The mounted DMG passed real-model Metal/headless API tests (ordinary/streaming,
seeded sampling, concurrent requests, overflow rejection, disconnect recovery,
shutdown). All model/file hashes and package source revisions were verified;
Windows executable version is 0.1.3.5; Linux executable bits were checked.
Android version 0.1.3/build 5, signature, ABI and ZIP alignment passed. All four
feedback audits passed. Physical Android Vulkan and clean-host Linux/Windows
real-model inference remain unqualified.

Artifacts: `logs/packages/release-0.1.3/`; evidence:
`logs/release-0.1.3-{ci,packaged-api,android-validation,linux-verify,windows-verify}.log`.
The GitHub release is still a draft while uploads complete.

### 0.1.3 publication completed

The draft status above is **superseded**: [DFM Mimir 0.1.3](https://github.com/schneiderkamplab/HRM-Text/releases/tag/dfm-mimir-v0.1.3)
was published on 2026-09-22 at 09:15:26 UTC as a public, non-prerelease, Latest
release. Tag `dfm-mimir-v0.1.3` resolves to package source `2137f43`. All four
packages and four checksum sidecars have matching GitHub SHA-256 digests and
sizes; the published notes match `native/app/releases/0.1.3.md` exactly.
Verification is recorded in `logs/release-0.1.3-publication.log`. Previous releases
and tags remain intact.

## 0.1.4 preparation: UCloud training credit — 2026-09-22

Unreleased source is now 0.1.4+6. The Flutter welcome screen pairs the DFM logo
with the user-supplied UCloud logo and the exact caption “Trained on SDU UCloud”.
The logo pair wraps into a vertical layout on narrow screens and uses a white
panel for contrast in either theme. The original SVG is tracked under
`native/app/branding/`; a 1200-pixel PNG with IBM Plex Sans rendered into it is
bundled offline. See [branding provenance](../../../native/app/branding/README.md).
No 0.1.4 release, tag or distribution upload is authorized yet; 0.1.3 remains
Latest.

Validation: the raster export was visually checked; Flutter analysis and all
52 existing tests pass, and OKF validation has no errors or warnings.

## Concise default model identity — 2026-09-23

The bundled Flutter profile, all three official catalog profiles, and Apple
profile/default now use:

> You are Mimir, trained on SDU UCloud by Danish Foundation Models, a Danish research collaboration developing open language models. You run locally on the user's device. Answer in the user's language.

The short DFM description follows its [official overview](https://www.foundationmodels.dk/).
This is part of unreleased 0.1.4. Existing saved/imported profiles retain their
own system prompts; this change updates the shipped defaults, not user data.
JSON parsing and agreement across all default profiles were checked.

## Search and saved prompt correction — 2026-09-23

See [optional Mimir search](mimir-search.md) for the unreleased 0.1.4 search
setting and service. Superseded: the earlier behavior retaining all old saved
stock prompts. Exact former defaults now adopt the research-collaboration prompt;
customized prompts remain unchanged. Reply completion restores composer focus.

## Default training languages — 2026-09-23

Supersedes the default wording in “Concise default model identity” above:

> You are Mimir, trained for Danish and English on SDU UCloud by Danish Foundation Models, a Danish research collaboration developing open language models. You run locally on the user's device. Answer in the user's language.

Updated the Flutter bundled/catalog profiles and Apple profile/default. Flutter
upgrades exact former stock prompts while preserving custom prompts. This source
change does not alter published packages.

## App text scaling (2026-09-24)

Flutter Settings offers 75–200% text sizing (5% steps, default/reset 100%).
`ChatStore.textScale` persists in the local archive independently of model
settings; invalid/missing values default to 100%. The root MediaQuery wraps the
Navigator, including open dialogs, and composes with nonlinear system scaling.
No native inference or network work is triggered. Native OS dialogs use system
sizing. Tests cover persistence, invalid values, nonlinear scaling and live
composer/dialog updates at phone and desktop widths with system enlargement.

Settings tabs: Model, Appearance, Online and Advanced. Advanced holds compaction
and profiles; Online holds search/feedback. Dependent controls hide when their
feature is off, preserving values; temporary busy locks stay disabled in place.
The model status describes acceleration; backend details sit by its selector.
