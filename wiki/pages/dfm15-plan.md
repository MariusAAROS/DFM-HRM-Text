---
type: Plan
title: DFM15 Native Reasoning
description: Native thinking and direct-answer supervision inherited from immutable DFM14.
status: draft
confidence: high
last_updated: 2026-10-15
tags: [dfm15, reasoning, data, training]
---
# DFM15

## Confirmed Dataset Additions

User decision on 2026-10-15: include both instruction datasets in DFM15:

| Language | Hugging Face dataset | Integration status |
|---|---|---|
| Faroese | `danish-foundation-models/faroese-dyna-instruct` | Downloaded, tokenized, registered; inherited FO-Instruct excluded |
| Icelandic | `danish-foundation-models/icelandic-dyna-instruct` | Downloaded, tokenized, registered |

Before rebuilding DFM15, inspect their schemas and source lineage, convert to
the native Gemma 4 conversation/reasoning format, check overlap with inherited
Dyna-derived examples, and tokenize. Determine sampling weights from measured
row/token counts; preparation defaults to repeat 1. The staged
token totals below do not include these additions. DFM14 remains unchanged.

### Organization Scan and Prepared Additions (2026-10-15)

The user authorized checking overlaps and integrating new organization
datasets/subsets. `scripts/scan_dfm15_organization.py` pins the 42 visible Hub
repositories and file hashes in `data/dfm15/organization-inventory.json`.
`config/dfm15_sources.json` registers these tokenized additions on top of the
native-reasoning staged base, with per-part selection indices:

| Source | Selected rows | Native Gemma tokens before length admission |
|---|---:|---:|
| Faroese dyna-instruct | 1,008,797 | 75,274,250 |
| Icelandic dyna-instruct | 610,104 | 100,239,190 |
| multilingual-gsm-symbolic, English synthetic train only | 3,799 | 682,844 |
| linguistic-quality, Danish correction pairs | 615 | 101,629 |

These are **tokenized staging**, not a rebuilt sampled DFM15. Full conversations
were preserved: 614 Faroese and 1,824 Icelandic sequences exceed 4,096 tokens;
future sample admission must handle them explicitly without truncating targets.
GSM-Symbolic uses native thought-channel reasoning and one boxed final answer;
21 exact internal duplicates were removed. Its 191 original training rows are
GSM8k-derived and not added again; no test split is admitted.

Overlap findings:

- All 571 composite FO-Instruct rows exactly match the pinned inherited
  `Setur/fo-instruct` multiset (NFC/outer-whitespace normalized). Preserve the
  inherited repeat-10 route and exclude the composite copy.
- No further identical normalized conversations occur across the four prepared
  additions. Evidence: `data/dfm15/overlap-report.json`.
- Danish and Norwegian dyna-instruct revisions and payload hashes are unchanged
  from the October 5 scan. Preserve earlier constituent exclusions, including
  duplicated direct Synquid sources and held-out Norwegian evaluation sources.
- This is **not** an exhaustive row-level match against the entire inherited
  corpus. Shared DynaWord grounding lineage is not proof of identical SFT rows.

Other organization repositories: retain existing audited routes for Laerebogen,
Arena, model-charter and Kaenguruen, existing Croco selection, and the explicit
danish-wildchat4.8M exclusion. Raw DynaWord/Gigaword/Nasjonalt vitenarkiv and
metadata collections are not direct instruction additions; creating new
grounded transformations from them is separate work. `Icelandic-flan` has no
Parquet/JSONL payload. Protect evaluation-only splits and existing DaLA holdouts.

Two potentially useful private collections remain blocked: `dialog-k2-da`
(89,545 filtered conversations; do not also add its unfiltered copy) and
`instruction-dataset-v1` (constituent and all-shuffled copies must not both be
included). The Hub returned **403: organization private repository storage
limit reached**, not an ordinary missing-token error. Their schemas can be
listed but content could not be downloaded. Once access is restored, compare
their constituents (Dialog, Dolly, EuroBlocks, FLAN, Laerebogen, OASST2,
OpenCodeInstruct, SciRIFF) against inherited routes before admitting increments.

Reproducible CPU commands:

```bash
python -m scripts.scan_dfm15_organization
python -m scripts.prepare_dfm15_dyna_instruct --workers 16
python -m scripts.prepare_dfm15_small_additions
python -m scripts.register_dfm15_additions
```

Sources: [Faroese](https://huggingface.co/datasets/danish-foundation-models/faroese-dyna-instruct),
[Icelandic](https://huggingface.co/datasets/danish-foundation-models/icelandic-dyna-instruct),
[GSM-Symbolic](https://huggingface.co/datasets/danish-foundation-models/multilingual-gsm-symbolic),
[linguistic quality](https://huggingface.co/datasets/danish-foundation-models/linguistic-quality).

## Evaluation Inheritance

Retain the complete [DFM14 evaluation baseline](dfm14-evaluation-baseline.md),
including all 50 languages and Andersen modernization. Native-reasoning
diagnostics are separately named additions, not replacements for existing
benchmarks or average populations; proposed additions are listed there.

## Version Boundary

User decision on 2026-10-09: native-reasoning repairs belong to **DFM15**.
DFM14 remains unchanged. Do not promote either the old plaintext repair or
the native-reasoning staging output into `data/sampled_dfm14`; do not change
ongoing DFM14 training or its scheduler as part of this data preparation.

The completed CPU build has been designated the first DFM15 staged base:

- Work root: `data/dfm15/native-reasoning-v2`.
- Sample: `data/dfm15/native-reasoning-v2/sampled_dfm15`.
- Three index sets: `epoch_0`, `epoch_1`, `epoch_2`.
- Mean tokens per epoch: **134,359,356,757** (134.359B).
- Native-thinking occurrences from repaired sources: about **2.079M per epoch**.
- Quarantined sampled occurrences removed: **38,828 / 38,976 / 38,996**.
- Status: validated staging, not promoted, not selected by live training.
- Original build log remains `logs/dfm14/native-reasoning-v2/build.log` as
  historical evidence; the build started before the DFM15 naming decision.

## Policy and Implementation

Preserve well-formed reasoning and final answers separately in native Gemma
channels with thinking enabled. Repair only unambiguous malformed boundaries;
retain an intact final answer in non-thinking mode when reasoning cannot safely
be retained or exceeds context. Quarantine examples without a complete usable
answer. Keep ordinary direct-answer supervision. Never truncate a thought
channel or remove its final answer merely to meet context length.

The implementation currently lives in `dfm14/native_reasoning.py` and
`dfm14/build_native_reasoning.py` because DFM14 is its immutable input. Its
default output is now DFM15. See the [repair details and history](dfm14-cpu-release.md)
for source coverage, tests, tool-specific repairs and evidence layout.
The stage carries its own template; shared/live templates remain untouched.

Before eventual training: finalize other DFM15 additions, review mode exposure,
select the required epoch index convention, calculate the packed-step budget,
and perform a checkpoint-safe explicit data switch. None of these are implied
by naming this staged corpus DFM15.
