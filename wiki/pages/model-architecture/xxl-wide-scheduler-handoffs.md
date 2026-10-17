---
type: Operational History
title: Local XXL-wide Scheduler Handoffs
description: Verified local SQLite migration, evaluation repairs, and cooperative CUDA suspension handoffs.
tags: [evaluation, scheduler, training, suspension]
status: stable
last_updated: 2026-10-17
confidence: high
---
# Local XXL-wide Scheduler Handoffs

Runtime guidance is in the [request-shared scheduler runbook](request-shared-eval-scheduler.md).

### 2026-10-17: Repair of the 1.1M path failures

The missing-path issue below was repaired without changing shared held-out pins.
`python -m scripts.localize_dfm14_eval` generates machine-local configuration
under `data/dfm14/evaluation/local-config`, verifying each selected/input file's
SHA256 before substituting paths. Shared configuration had pointed at the other
machine's `/work/mimir/HRM-Text` tree. Future pending rows now use the localized
configuration too.

Training cooperatively suspended at 1,118,800 (generation 6), wrote its fallback
checkpoint, and offloaded all eight workers. All 128 failed inference shards
then completed successfully, as did their task merges. Final averaging was
still running at verification; training restoration was not yet confirmed.
All nine checkpoint average jobs were reset to depend on the repaired merges,
avoiding publication of premature incomplete averages. The terminal recovery
gate permits training continuation even if finalization fails.

`scripts/repair_1100k_eval_handoff.py` records the one-off machine-specific
recovery, not a reusable resume command. Do not replay it against the live plan;
it requires explicit `--execute`. SQLite is authoritative, with plan.tsv exported
as a snapshot after edits. Use the cooperative supervisor, not a second torchrun.

### 2026-10-16: 1.1M evaluation handoff

Verified automatic generation-5 restoration at step 1,100,000, followed by
training progress through 1,100,020 on all eight GPUs. The
`xxlw-dfm14-train-1150000-evals-terminal` gate completed with 919 successful
and 128 failed inference dependencies; the 1,150K training boundary is running.
Failed DFM shards (an inspected DaLA shard references missing
`/work/mimir/HRM-Text/evaluation`) and blocked publication rows did not prevent
training continuation. These failures still need separate repair; the checkpoint
evaluation is not complete. No manual restore or plan change was necessary.
Post-resume step deltas were about 2.9 seconds/step; tqdm's smoothed rate includes
the suspension and is not a valid immediate training-speed estimate.

### 2026-10-12: Early 1M supplement handoff

The requested suspension at 1,001,250 arrived after that step; the training
hook accepted it at **1,001,270**. All eight ranks published readiness after
writing `checkpoint_state_step_1001270.json`, then CUDA checkpointing offloaded
their GPU state. The existing launcher/supervisor and worker processes survive.

For `logs/scheduler/dfm10_XL_epoch9_20260831`, the SQLite plan now places the
403 missing 1M inference jobs before resuming toward 1,050,000. The prior
644 inference results are preserved. An explicit completed boundary
`xxlw-dfm14-train-early-1001270` records the verified offload; the existing
1050K training row remains pending, not falsely completed.
`xxlw-dfm14-1000000-supplement-evals-terminal` waits only for those 403
inference jobs to become terminal, including exhausted failures. Merges,
averages, and publication do not block restoration. The existing training row
drains owned evaluation servers and requests the sole supervisor to restore;
its disk fallback now starts from `step_1001270` if the launcher is lost.

The old coordinator alone was stopped by verified PID after offloading; no
training workers or supervisor were killed. The replacement uses the same
environment and request-shared SQLite settings. Administrative receipt:
`early-1001270-handoff.json`; backup: `before-early-1001270.sqlite`; runner log:
`runner-early-1001270.log`, all in that plan directory. The Rich monitor follows
the same plan. Restoration after these evaluations is scheduled, not yet
observed at the time of this entry.

Follow-up verification (2026-10-13): restoration succeeded in the same eight
worker PIDs (397377, 397378, 397381, 397383, 397387, 397394, 397396, 397397),
with matching generation-1 `resumed.json`. Training passed step 1,002,120 at
approximately 2.85 seconds/step and wrote the 1,002,000 ephemeral. No evaluation
GPU processes remained; no disk fallback was needed. This verifies operational
resume, not numerical parity. The evaluation interval was only partially
successful: 160 supplemental DFM inference jobs completed, while all 243
supplemental EuroEval jobs failed with `KeyError: 'hf_export_dir'`. Those jobs
still need configuration repair and rerunning; training correctly resumed
despite their terminal failures. Do not describe the 1M supplements as complete.

The subsequent repair adds `hf_export_dir` from the local model path to all
4,374 pending/failed supplemental EuroEval inference/merge definitions across
the nine checkpoints. The proxy now also resolves the existing
`hrm_hf_export_dir`/`standard_hf_export_dir` keys, and the harmonization builder
supplies the explicit field. Eight focused regression tests pass. Training
suspended at 1,002,170 (generation 2), with verified disk checkpoint and GPU
offload. Only the 243 failed EuroEval inference jobs were reset; the successful
160 DFM jobs remain done. The new `xxlw-dfm14-1000000-euro-repair-terminal`
gate is inference-only, and the existing 1050K training row resumes from the
same process or `step_1002170` disk fallback. Receipt and runner log:
`euroeval-repair-1002170.json`, `runner-euro-repair-1002170.log`. This retry's
eventual restoration remains to be verified; do not infer success from dispatch.

Verified outcome on 2026-10-13: all 243 EuroEval retries succeeded, completing
all **1,047** 1M inference jobs. The scheduler drained its eight vLLM servers
and the supervisor restored generation 2 into the same eight training PIDs.
`resumed.json` matches step 1,002,170; no disk fallback was used. After roughly
103 minutes suspended, training passed 1,002,200. Log timestamps show 25 steps
in 72 seconds (2.88 s/step), consistent with the prior rate. The immediate tqdm
smoothed ETA includes the suspension and is misleading until it settles.
Seven post-processing rows remained pending at this verification; inference
completion does not imply that all publication/averaging is finished.

Performance follow-up: direct-mode startup admitted many clients onto the
first ready replica while the other seven were provisioning. At one point
57 clients resided on GPU0 and one on each other GPU; the eventual tail had
41 clients on GPU0 alone. They continued progressing and were not restarted.
Review startup admission balance before the next campaign; this is separate
from the repaired tokenizer-path failure and from successful CUDA restoration.

Later user-requested short pause (2026-10-13): training suspended at
**1,015,800**, saved its regular fallback checkpoint, and successfully offloaded
all eight ranks (no compute processes listed by nvidia-smi). The supervisor-root
`hold` and SQLite administrative dispatch lock remain set pending explicit user
resume. The existing 1050K boundary row remains running/waiting, not completed.
To resume this manual pause, use the current active control directory's matching
generation restore request through the existing supervisor, remove the hold,
and clear the SQLite admin lock; do not start a second training launcher.

The user subsequently requested resume: generation 3 restored successfully
into the same eight worker PIDs and training passed 1,015,810. The supervisor
hold and dispatch lock are cleared. No disk fallback or second launcher was
needed; five post-resume steps took 14 seconds.

### 2026-10-10: XXL-wide /work/dfm migration at 950K

This is the other machine's XXL-wide campaign, **not** the XL campaign below.
The five scheduler commits are already in local `main` at `089550e`.
Existing plan: `logs/scheduler/dfm10_XL_epoch9_20260831`.
Legacy runner PID 103759 has a soft stop request; the active
`xxlw-dfm14-train-950000` segment continues normally. Do not clear that request
or start a second runner while migration is waiting.

`scripts/migrate_xxl_wide_sqlite.py` is detached (initial PID 3074897), waiting
for that runner to drain. It refuses running rows, failed boundary training,
an incomplete checkpoint, or an existing database requiring recovery. It backs
up the current TSV, rechecks preflight, imports the same campaign, backs up
SQLite, clears the stop, and launches the new runner. Completed rows are not
replayed. Migration is **scheduled, not yet observed complete**; verify
`sqlite-migration.json`, `runtime.json`, and logs before reporting it completed.

- Authority: `/tmp/dfm14-xxl-wide-production/jobs.sqlite` (local overlay).
  `/work/dfm` is WekaFS and must only hold snapshots/backups, not SQLite WAL.
- Runner: `hrm`, `--backend sqlite --persistent-vllm --max-clients 64
  --server-concurrency 1024 --cpu-cores 256 --reserve-cores 8`, GPUs 0-7.
- CUDA 13.2 compiler verified under
  `/home/ucloud/miniforge3/envs/hrm/lib/python3.13/site-packages/nvidia/cu13`;
  set `CUDA_HOME` there and prepend its `bin` plus `hrm/bin` to PATH.
- Preserve local target allocations: 0.9 normal, 0.65 judged; retain
  `unsloth/gemma-4-E4B-it` and its existing judge engine/settings. The other
  machine's XL allocations are not automatically appropriate for XXL-wide.
- Ten judge-first phase gates and nine failed-eval-tolerant training handoffs
  were explicitly applied to pending DFM14 rows. Model, scorer, batch sizes,
  W&B IDs, training commands and historical runtime states were preserved.
  Revised preflight: 4,538 pending rows, zero failures.
- The XXL-wide wrapper `dfm14.continue_training.finalize_boundary` now publishes
  exact cursor-derived `eval_epoch` **and** `eval_step` in a SQLite transaction.
  It preserves running wait tokens, refuses active inference, and retains the
  TSV path. Focused wrapper/store/grouping/runner tests: 77 passed.
- Rich monitor: existing tmux pane `%2` (shared window 4), refresh 30 seconds.
  The migration only replaces it if it is still an eval monitor.

Watch `logs/scheduler/dfm10_XL_epoch9_20260831/sqlite-migration.log` for handoff;
then `runner-sqlite.log` in that directory. The authoritative database is
periodically backed up to `state.backup.sqlite` there. After migration use
database editing/recovery commands, not direct writes to the old `plan.tsv`.
DFM14 data and the DFM15 native-reasoning staging remain unchanged.

**Observed completion later on 2026-10-10:** the 950K training segment and
export completed. Migration finished at 21:58 local time; SQLite runner PID
3393903 started with 64 clients and split 2,850 publication phases from
inference. All eight judged shards completed and merged before non-judged
work began. The Rich monitor is live in window 4. This supersedes the
"not yet observed complete" status above.

The user's subsequent coverage check found a pre-existing plan omission:
only Danish/English legacy tasks and the 16 DFM14 languages were scheduled,
not the earlier 34 languages' accepted-v2 tasks. At the user's request,
`scripts/prepare_xxl_all_language_evals.py` recovered all 34 selections from
`schneiderkamplab/dfm13-dala-v2-{language}-compact` test-representative files.
Both published input hashes and the reconstructed 1,000-pair selected-file
hash must match the original committed manifests exactly. No train rows or
new test selections are substituted. Remote build receipt references remain
in recovery provenance; runtime pins use the verified published test files.
Ports 2850 and 6768 were unreachable during this recovery.

Local inputs/configs and `ready.json` are under
`data/dfm14/evaluation/all50`. Recovery uses `hrm`; constructor verification
uses `dfm-evals/.venv/bin/python` with `PYTHONPATH=.:dfm-evals`, because
`inspect_ai` is installed in the eval venv, not base `hrm`:

```bash
PYTHONPATH=.:dfm-evals dfm-evals/.venv/bin/python \
  scripts/prepare_xxl_all_language_evals.py --verify-only
```

`scripts/extend_xxl_all_language_evals.py` applies the verified registry under
the database admin claim lock, preserving running tokens/completions. The
2026-10-10 install added 6,120 rows over ten checkpoints (950K through final
1,391,204): 68 tasks/checkpoint, four inference shards and four CPU export
rows plus one merge per task. DaLA and GEC now each cover all 50 languages,
2,000 examples per task. Existing Danish/English benchmarks remain unchanged;
this is **not** a claim of EuroEval coverage in every language. Multilingual
averages wait for the additional merges, and terminal training barriers also
wait for them while retaining failure-tolerant continuation. The prior
multilingual manifests still report coverage gaps for unrun legacy/EuroEval
metric slots rather than inventing values. Four migration/extension tests
passed; all 272 new shard constructors passed. The extension unlocked claims
and new-language clients started successfully.

Comparison with the live remote plan on SSH port **2103** (its pending 3450K
checkpoint, excluding superseded/skipped rows) confirms that language coverage
is not full campaign parity. Local/remote unique task counts: standard 8/8,
DFM excluding IFEval 131/170, EuroEval including IFEval 20/261; averages 2/9.
The local plan lacks 243 remote EuroEval tasks and 40 historical DaLA/GEC tasks
that the remote retains alongside v2. Local-only tasks are Andersen
modernization and VALEU-da/en. Common inference shard counts agree. Both log
to DFM5, local `dfm10-xxl-wide` versus remote
`dfm8-xl-from-dfm6-dfm7-epoch5-clean-full`; model-specific capacity settings
must not be copied blindly. Read-only snapshots:
`logs/diagnostics/scheduler-parity-20261010/{local-950k,remote-3450k}.json`.

**2026-10-12 update:** The local 1M-and-later campaign gap above is now
reconciled against port 2546, including nine average jobs. See the
[deployment receipt and deferred 1M supplementation](../dfm14-evaluation-baseline.md#2026-10-12-local-xxl-campaign-reconciliation-installed).
Historical 950K and completed dependency versions are not retroactively changed.
