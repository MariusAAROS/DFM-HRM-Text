---
type: Technical Reference
title: App Model Catalog and Release History
description: Model discovery, tokenizer corrections and 0.1.2 publication history.
tags: [mimir, app, releases]
status: draft
last_updated: 2026-10-17
confidence: high
---
# App Model Catalog and Release History

Part of [DFM Mimir app and portable backends](mimir-app.md).

## 2026-09-21: optional model library and public checkpoint inventory

The portable client now has a model selector with installed/imported history,
explicit selection, unselected-file removal, pinned HF downloads with progress,
SHA-256/size/GGUF checks and cancellation. Model-network permission is independent
of feedback consent and defaults off. Startup never fetches the catalog. Android
selection remains unloaded until the user explicitly starts the engine.

[Model inventory and maintainer workflow](../../../native/app/MODELS.md) documents
seven public HF repositories, current bundled Q4_K_M provenance, conversion needs
and model-specific catalog profiles. Catalog refresh from the repository branch
can add supported newer GGUFs without rebuilding the client. HF-discovered
repositories remain separate from catalog-approved download entries; names alone
cannot establish architecture/template compatibility. This is portable UI work;
the separate SwiftUI UI and headless CLI were not extended with a selector.

A real noctrex Q8_0 download passed its pinned hash and a Danish generation smoke
check on macOS Metal. Its rendered chat templates matched our qualified export
on four prompts, but token IDs matched only two: ` ø` and ` Hvad` examples differ.
Therefore these third-party catalog entries are explicitly experimental. Recommended
alternative precisions still need publication of our qualified GGUFs. This adds
concrete evidence to the earlier policy of using our own tokenizer-corrected exports.

Static analysis and 37 app tests pass; the ten new tests cover offline permission,
selector UI, discovery separation, valid download, hash/size/magic rejection,
mid-stream revocation, pinned metadata and persisted Android selection/removal.
The macOS release and iOS simulator builds pass. Download retries restart from zero; background
OS transfers are not implemented. See the linked document for full limitations.


### Official GGUF discovery extension, 2026-09-21

The earlier repository-only discovery policy above is **superseded** for
`danish-foundation-models`: catalog refresh now merges curated entries with all
GGUF files in official repositories whose names contain `mimir` case-insensitively.
It follows repository/file pagination and pins revision/size/LFS SHA-256. Curated
profiles take precedence for duplicates; auto-discovered models use an explicitly
unqualified profile and select with 1024 context/512 reply, automatic sizing off.
Unsupported shards and missing-integrity files remain visible with reasons.
Network consent is unchanged; startup stays offline. Independent source failures
preserve cached entries and report errors. The official namespace currently has
one matching repository and no GGUF files (live API check).

[Catalog update instructions](../../../native/app/MODELS.md#adding-models-without-rebuilding-the-app)
explain direct JSON edits/pushes and the no-edit path of uploading official GGUFs.

Validation: static analysis clean and all 41 app tests pass, including official
case-insensitive matching, nested file/repository pagination, pinned metadata,
curated precedence, partial-source failure recovery and unsupported-file listing.


Follow-up inspection of the public noctrex Q8_0 and bundled Q4_K_M GGUF headers
confirmed the tokenizer mismatch: public Q8_0 uses `gemma4` pre-tokenization with
zero byte-fallback token types, while the qualified export uses
`spm-bpe-mistral` with 256 byte-fallback tokens, matching the original checkpoint's
`fix_mistral_regex=true` configuration. This is independent of weight precision;
other public GGUF precisions have not been independently checked.


### Training tokenizer reference correction, 2026-09-21

**Supersedes the inference above that the public `gemma4` pre-tokenizer is wrong.**
The training path loads raw `tokenizer.json` using `tokenizers.Tokenizer.from_file`
and encodes templated prompts directly; training consumes the resulting stored
IDs. `fix_mistral_regex=true` is injected later by the HF exporter, so it is not
a training specification. Reproducing the training loader with the pinned HF
JSON (byte-identical to local `dfm67_1150k/tokenizer.json`) matches public noctrex
Q8_0 on 19/19 cases and our bundled Q4_K_M on 15/19. Thus noctrex's **pre-tokenization**
is the match to the local training pipeline. Other tokenizer/decoder metadata and
weight quality still require independent qualification; differing byte-token type
counts alone do not establish which complete conversion is correct.

[Full evidence](../../../native/mimir/training-tokenizer-audit.json) records the
cases and token IDs; [app model notes](../../../native/app/MODELS.md#training-reference-correction-2026-09-21)
record provenance limits and next steps. The historical external tokenizer path
and original production tokenized corpus are absent locally, so this is a pipeline
reproduction against the exported JSON, not a complete corpus audit. No model,
exporter behavior or release asset was replaced by this investigation.


### Corrected 4/8/16-bit artifacts prepared, 2026-09-21

[Corrected GGUF preparation](../../../native/mimir/CORRECTED-GGUFS.md) supersedes
the prior statement that regeneration was pending. Fresh BF16, Q8_0 and Q4_K_M
artifacts are in `logs/mimir-corrected`, with checksums, licenses and provenance.
The llama.cpp HRM converter now loads the raw training graph (commit `4122b9a81`);
other architectures retain their original tokenizer loader. Native HF reference
generators likewise use the direct graph. Each artifact passes 724 tokenizer/
decoding/template cases, 19 independent training-audit cases, and CPU/Metal
Danish/English chat smoke tests. BF16's 259 tensors are bit-identical to the prior
BF16 export. Full quality/backend/device requalification remains distinct from
these bounded checks. Existing app assets, release packages and HF repositories
have not been replaced; these files are prepared locally for subsequent publication.


### Official GGUF publication completed, 2026-09-21

The local-only status above is **superseded**: corrected BF16, Q8_0 and Q4_K_M
files are public at [DFM-Mimir-GGUF](https://huggingface.co/danish-foundation-models/DFM-Mimir-GGUF),
revision `4364aaa61279c187b7cd202375915ea962b79afb`. Anonymous HF model-info queries
verified all three sizes and LFS SHA-256 values. The single publication commit
includes the model card (HF metadata validation passed), license, checksums,
provenance and bounded validation results. The uploader used existing org-admin
credentials; no credentials were copied into the upload folder or app assets.

The app catalog now pins these three official files ahead of the retained third-party
entries, with model-specific profiles. Automatic discovery also finds this new
Mimir repository. [Publication record](../../../native/mimir/hf-gguf/publication.json)
retains revision and file hashes; [model card source](../../../native/mimir/hf-gguf/README.md)
is versioned in this repo. No preview application packages or bundled app asset
were rebuilt/replaced in this publication step.

Publication/catalog validation: static analysis clean and all 41 app tests pass
with the expanded six-entry curated catalog.


### Official-only model catalog policy, 2026-09-21

The previous retained-third-party and Mimir-only discovery policies are
**superseded**. Both curated catalog loading (embedded, remote and cached) and
HF discovery require owner `danish-foundation-models` and repository names
containing both `mimir` and `gguf`, case-insensitively. The embedded catalog now
contains only the three corrected official exports. Cached discovery listings
are filtered at startup, and direct downloads enforce the same repository rule.
Already installed models and manual imports remain usable; networking is still
optional. No app packages were rebuilt for this source/catalog change.

Validation: Flutter analysis clean; all 44 app tests pass, including repository
filtering, offline cache restoration and blocked direct downloads. OKF validation
still reports the two pre-existing benchmark-charts index/frontmatter errors.


### Explicit third-party repository selection, 2026-09-21

The official-only policy now has a user-controlled exception: Models accepts
additional public HF `owner/repository` IDs, persists them, and discovers GGUFs
from those exact repositories on explicit refresh. IDs need neither name keyword.
Automatic official discovery and remote curated catalogs retain their existing
restriction; a remote entry cannot grant itself permission. Cached entries and
download admission accept the user's saved exceptions. Removing an exception
removes its listings but preserves installed files. Adding an ID does not enable
networking. Discovered third-party artifacts retain checksum/revision checks,
split-file restrictions and conservative, unverified model profiles.
See [model library usage](../../../native/app/MODELS.md#user-specified-hugging-face-repositories-2026-09-21).

Validation: all 46 app tests pass and Flutter analysis is clean. Coverage includes
UI add/remove, offline persistence of IDs and cached models, invalid IDs,
case-insensitive deduplication, verified downloads, failed lookup fallback and
revoked download access. The two existing benchmark-charts OKF errors remain.


### 0.1.2 package preparation, 2026-09-21

Portable app source is 0.1.2+4. The bundled-model policy supersedes 0.1.1:
use corrected official Q4_K_M SHA-256
`8c811ca72589112f00306e8b3a23f83944751d13d3fce6dfdb39df8b31726e49`
from HF `danish-foundation-models/DFM-Mimir-GGUF` revision
`4364aaa61279c187b7cd202375915ea962b79afb`. Includes model selection,
official discovery and user-specified third-party repositories. Packaging and
publication verification are pending below.


0.1.2 validation completed locally and in
[Linux/Windows CI 35585590918](https://github.com/schneiderkamplab/HRM-Text/actions/runs/35585590918)
for source `33875dcdee09736c6e987dd5dc9d5ddc51333ec2`: all 46 app tests and
analysis passed; CI also passed native fallback/compaction tests. macOS ARM64,
Android ARM64 and iOS simulator builds passed. The mounted DMG passed image,
signature and corrected-model checks, plus real-model Metal/headless API tests
(normal/streaming, seeds, concurrency, overflow, disconnect recovery, shutdown).
Linux/Windows bundles have verified per-file/model hashes and source revisions;
Linux executable permissions and Windows 0.1.2.4 executable version were checked.
All four packages passed feedback audits; Android signature, ARM64 packaging and
16 KiB ZIP alignment passed. No physical Android Vulkan or Linux/Windows
clean-host real-model/GPU qualification is claimed. Packaging macOS requires an
absolute `--dart` path because server compilation changes its working directory.

Local artifacts: `logs/packages/release-0.1.2/`; evidence is in
`logs/release-0.1.2-{tests,analyze,packaged-api,archives-verify,package-audits}.log`.


### 0.1.2 publication completed, 2026-09-21

The pending-publication status above is **superseded**. All four bundled-weight
packages and SHA-256 sidecars are public in
[DFM Mimir 0.1.2](https://github.com/schneiderkamplab/HRM-Text/releases/tag/dfm-mimir-v0.1.2).
Tag `dfm-mimir-v0.1.2` resolves to package source
`33875dcdee09736c6e987dd5dc9d5ddc51333ec2`; the release is public, non-prerelease
and marked latest. GitHub asset sizes and SHA-256 digests matched all eight local
files. The earlier 0.1.0/0.1.1 downloads remain available in the previous release.

A laptop network/DNS interruption terminated the first Linux/Windows upload;
GitHub release uploads required restarting those whole files. The retry succeeded.
macOS/Android were briefly published first with explicit pending notes; final
notes now list all four completed downloads and validation evidence. Publication
verification is recorded in `logs/release-0.1.2-publication.log` and
`logs/release-0.1.2-checksums.txt`.

```text
cc7b659eb653eb85820d46e5c320cf2a5e430e3646b22ddfac84e7b5b668b9c1  dfm-mimir-0.1.2-macos-arm64.dmg
8f7980cbc4cdd4d8e58fc0eb32ad3659bf2ca2746fd82cea4dd7684ec2222378  dfm-mimir-0.1.2-android-arm64.apk
5c1b0cdb3bbc24d7c2c976d288fa3efcf29325e04859064578762d8702871098  dfm-mimir-0.1.2-linux-x64.tar.gz
85cd9b17651a6a9018976a6b475120ed626ca82d8f580bc334954c50477af744  dfm-mimir-0.1.2-windows-x64.zip
```


### Separate release history, 2026-09-21

The earlier combined 0.1.0/0.1.1 release layout is **superseded**. There are now
three separate public, non-prerelease releases, each containing only its own four
platform packages and four SHA-256 sidecars:

- [0.1.0](https://github.com/schneiderkamplab/HRM-Text/releases/tag/dfm-mimir-v0.1.0):
  existing tag `7d56a56c5e5226d4db0ea961ef7fd2cd590395cf` retained; original
  0.1.0 title/notes restored and all 0.1.0 asset sizes/digests unchanged.
- [0.1.1](https://github.com/schneiderkamplab/HRM-Text/releases/tag/dfm-mimir-v0.1.1):
  new tag at recorded package source `1fa33a1c5656d1b288a13a04a1b1ee77f54833ff`.
  All eight assets were copied and verified against their original GitHub
  sizes/SHA-256 digests before removing them from the 0.1.0 release. No rebuild.
- [0.1.2](https://github.com/schneiderkamplab/HRM-Text/releases/tag/dfm-mimir-v0.1.2):
  tag `33875dcdee09736c6e987dd5dc9d5ddc51333ec2` retained; remains latest.

0.1.1 download URLs now use the 0.1.1 tag; its old asset URLs under the 0.1.0 tag
no longer apply. Historical combined-release links elsewhere on this page describe
the previous arrangement. Verification/backup records:
`logs/release-separation/{before,verified}.json` and `finish.log`.


### Self-contained release descriptions, 2026-09-21

Release descriptions must explain the shipped app without relying on earlier
release notes. The 0.1.2 description is now maintained in
[the versioned release document](../../../native/app/releases/0.1.2.md), with
[maintenance guidance](../../../native/app/releases/README.md). It includes
installation/backend/signing limits; model selection and network consent;
context/compaction/MixedLM; desktop API settings, address/authentication, requests
and limitations; per-platform headless commands; feedback consent/publication
choices; and preserved provenance, validation and checksums. Examples/UI labels
were checked against the shipped source and API guide. Publishing this description
does not change any package, checksum, tag or release status.


### Release-description policy for all versions, 2026-09-21

The self-contained description requirement applies to every future patch, minor
and major release, including 0.1.3+, 0.2.x and later. The
[release policy](../../../native/app/releases/README.md),
[coverage template](../../../native/app/releases/TEMPLATE.md), and
[packaging plan](../../../native/app/PACKAGING-PLAN.md#release-descriptions-for-every-version)
make this a preparation/publishing requirement, not a promise to publish future
versions automatically. Maintain each exact body in `native/app/releases/VERSION.md`
and verify it against that release's source and artifacts, including unchanged
features and restrictions.

[0.1.1's complete description](../../../native/app/releases/0.1.1.md) now covers
installation, local GGUF/profile import, context/compaction/MixedLM, API setup and
examples, headless commands, feedback/privacy and original validation/checksums.
It was checked against source `1fa33a1`; it does not claim the later HF catalog or
corrected bundled tokenizer. Updating the GitHub description preserves release
assets, checksums, tag target and visibility.

