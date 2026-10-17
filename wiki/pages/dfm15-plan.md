---
type: Plan
title: DFM15 Native Reasoning
description: Native thinking and direct-answer supervision inherited from immutable DFM14.
status: draft
confidence: high
last_updated: 2026-10-17
tags: [dfm15, reasoning, data, training]
---
# DFM15

## Math Conditioning Handoff (2026-10-17)

The earlier full-mixture hold release is superseded for the newly approved
conservative math-format conditioning. The DFM15 training row is pending with
`dfm15_preparation_hold=true` and an unreachable GPU-memory admission threshold.
Existing evaluations continue. Both the hold-release helper and training launcher
require `data/sampled_dfm15/math-answer-conditioning-verified.json` before
training may start; the full-mixture receipt alone is no longer sufficient.
Only successful conditioning and verification may release this hold.

CPU conditioning was launched on 2026-10-17 after two focused converter tests
passed (mode preservation and conservative exclusions). It uses 16 conversion
workers after collecting source boundaries and scanning the three index sets.
Progress log: `logs/training/dfm15_XL_extension/math-conditioning.log`.
No GPU work or benchmark scorer changes are involved.

Conditioning completed and all three index sets were verified on 2026-10-17.
Exactly 1,250 unique targets (1,250 occurrences per epoch) were changed; the
conservative matcher left 743,095 cohort candidates without an unambiguous
numeric final unchanged. This is limited format exposure, not a corpus-wide
math normalization. The verification receipt was published and the training
hold released; normal evaluation dependencies still govern training admission.

The 1,250-example intervention above is superseded by a second pass requested
on the same day. Inspection found that the first matcher rejected terminal
`\\boxed{25}.` because of punctuation and bare direct numeric targets such as
`3`. It also confused ordinary problem wording such as "only three apples"
with output constraints. Version 2 handles these cases, keeps the same fixed
10% source-offset cohort, and preserves symbolic/ambiguous targets and genuine
format constraints. It preserves worked solutions verbatim and appends the
requested `ANSWER:` line; direct numeric targets receive that prefix without
adding reasoning. Existing native thinking channels are retained.

Version 2 uses `data/dfm15/math-answer-conditioning-v2` for evidence and
`logs/training/dfm15_XL_extension/math-conditioning-v2.log` for progress.
The scheduler hold was reinstated; the launcher and release helper now require
a version-2 receipt. Six focused tests passed. Index replacements are staged
one epoch at a time with a commit journal, rather than changing four live
arrays independently without recovery. Mathematical answer correctness is
inherited from the sources; this format pass does not independently solve them.

Version-2 preparation selected 521,883 additional unique examples: 521,698
OpenMathInstruct2, 159 AllenAI Open Math, 25 GSM-Symbolic and one Tulu personas
math example. Alongside version 1 this gives 523,133 conditioned examples per
epoch, about 6.7% of the scoped math-source occurrences. The net version-2
addition is 11,272,888 tokens per epoch. Seven focused tests passed, including
staged publication, preservation of unchanged rows and token accounting.
A seeded 1,000-row decoded check confirmed prompt/target transformations,
thinking-channel counts and the 4097-token limit.

Quality caveat discovered during inspection: the changed Tulu example asks
about products of pass totals, while its inherited final answer claims 18
successful goals. The format pass preserves that existing answer; it does not
establish correctness or repair source-level semantic inconsistencies. Source
evidence is retained in `math-answer-conditioning-v2/examples.jsonl`.

Final verification completed for all three epochs; temporary staged index
arrays and variant token buffers were removed. Mean DFM15 size is now
**134,516,139,170 tokens/epoch**. The version-2 receipt was published and the
DFM15 preparation hold released, with existing evaluation dependencies intact.

## Full Local Sample and Scorer Compatibility (2026-10-17)

Upstream pulled through `0c9fe65`. The repair-only local sample was replaced
in place using `scripts/build_dfm15_full_sample.py`; its large token backing
was reused, not copied. Only full `data/sampled_dfm15` remains. All three
index sets pass bounds, nonempty-target, length and token-total verification.
Per epoch additions at repeat 1 after whole-row length admission:

| Addition | Rows | Tokens |
|---|---:|---:|
| Faroese Dyna-Instruct | 1,008,183 | 70,610,198 |
| Icelandic Dyna-Instruct | 608,280 | 74,094,689 |
| English GSM-Symbolic synthetic train | 3,799 | 682,844 |
| Danish linguistic-quality corrections | 615 | 101,629 |
| Total additions | 1,620,877 | 145,489,360 |

Mean full-mixture tokens/epoch: **134,504,846,117**. Omitted 614 Faroese and
1,824 Icelandic overlength conversations, without truncating targets.
Inherited selection/repetition is retained, with additions uniformly
interleaved at deterministic positions per epoch. Exact duplicate FO-Instruct
remains excluded from the composite addition, retaining the inherited route.
Publication receipt: `data/sampled_dfm15/full-mixture-verified.json` pins
`config/dfm15_sources.json`. The existing scheduler training hold was released
only after this receipt was published. The continuation wrapper requires it.

GSM-Symbolic conversion verifies `####` against the numeric target, removes
calculator markup, puts reasoning in native Gemma thought channels, and emits
one `\\boxed{number}` final answer. Decoded tokenized row 0 has thought
`78+39 = 117` and final `\\boxed{117}`. This matches our GSM8k extractor
(`evaluation/benchmarks.py`) and both projects' boxed-first MATH scoring.

**HRM-test distinction:** on `ssh.cloud.sdu.dk:6768`,
`/work/dfm/HRM-test/run_three_model_queue.py` selects built-in
`inspect_evals/gsm8k`, not its older `custom_english_tasks.py@gsm8k_0shot`.
The installed built-in uses 10 shots, a step-by-step `ANSWER: ...` prompt,
and `match(numeric=True)` from the brainsurgery environment. Direct CPU tests
of its installed matching functions reject `\\boxed{117}` and accept
`ANSWER: 117`; a preceding standalone `39` can become the extracted answer
when the final box cannot be parsed. The older custom scorer explicitly
accepts boxes. HRM-test's MATH custom/HF scorers extract a final box and use
math_verify; its IE MATH scorer tries `ANSWER:` first, then a box, then full
text. Its IE prompt requests `ANSWER:` even though boxes score correctly.
No scorer or target-format changes were made as part of this inspection.

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

## XL Extension Approved 2026-10-16

2026-10-17 recovery: DFM14 completed step 3,768,319 at 03:25 CEST,
but finalization failed because `display_epoch` read an in-epoch row cursor
from an epoch checkpoint. Epoch checkpoints legitimately omit that field.
Final checkpoints now map directly to parent epoch + 1; intermediate
checkpoints still require and validate the cursor. Regression-tested with a
cursor-free epoch sidecar. Resetting the failed training row also rearms the
terminal inference barriers, so DFM15 cannot bypass the final DFM14 evals.
DFM15 transfer and validation completed and its continuation was installed
at 00:50 CEST; data readiness was not the cause of this stall.

**Superseded 2026-10-17:** selecting the native-reasoning base alone was incorrect.
The continuation requires all four registered additions from
`config/dfm15_sources.json` as well as the repaired inherited mixture.
Reuse the transferred backing in place; do not retain a second repair-only sample.
Continue the existing XL run after DFM14 epoch completion at
3,768,319 through step 3,800,000 (31,681 steps). Preserve optimizer and EMA;
reset only the data cursor to DFM15 `epoch_0` using an isolated checkpoint
wrapper. Keep base LR 1e-5 with lr_auto, BP8, GAS2, no warmup or decay.
The training wrapper requires a full-mixture validation receipt before launch.

`scripts/prepare_dfm15_extension.py` transfers the validated remote sample,
checks tokenizer identity, index bounds/lengths and token totals for all three
index sets, and atomically publishes `data/sampled_dfm15`.
`scripts/schedule_dfm15_xl_extension.py --wait` installs the continuation into
the existing authoritative SQLite plan only after validation. Training follows
successful DFM14 completion and terminal final-epoch GPU inference, not CPU
publication. Same W&B run: `DFM5/dfm8-xl-from-dfm6-dfm7-epoch5-clean-full`.
Logs and installation receipt: `logs/training/dfm15_XL_extension/`.

The archive watcher now requires the 3.8M checkpoint as well as completed
DFM14 final evaluations before final catch-up. It includes the DFM15 sample
and checkpoints and writes a remote completion receipt. The initial archive
remains resumable and runs concurrently; do not shut down before its final
receipt is present.

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
