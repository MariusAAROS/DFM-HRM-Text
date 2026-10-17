"""Clone the final DFM14 campaign, preserving production settings and dependencies."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval_scheduler'))
from eval_scheduler.model import Action, JobStatus
from eval_scheduler.store import SQLiteJobStore
from eval_scheduler.database_runner import CLIENTS

DB = '/tmp/dfm14-production-3383000/jobs.sqlite'
PREFIX = 'dfm15-xl-3800000-eval-'
TRAIN = 'dfm15-xl-step_3800000-train'
PLAN = ROOT / 'logs/scheduler/dfm12_XL_epoch11_noidentity'


def rewrite(value):
    if isinstance(value, dict):
        return {k: rewrite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite(v) for v in value]
    if isinstance(value, str):
        for old, new in [
            ('checkpoints/dfm14/XL-from-dfm13-step3250000', 'checkpoints/dfm15/XL-from-dfm14-epoch1'),
            ('dfm14_XL_epoch_1_ema_hf', 'dfm15_XL_step_3800000_ema_hf'),
            ('dfm14_XL', 'dfm15_XL'), ('hrm-dfm14-XL', 'hrm-dfm15-XL'),
            ('/epoch_1', '/step_3800000'), ('3768319', '3800000')]:
            value = value.replace(old, new)
        return 'step_3800000' if value == 'epoch_1' else value
    return value


def install():
    store = SQLiteJobStore(DB)
    if store.get_admin_lock():
        raise RuntimeError('Another administrator holds the plan lock')
    store.set_admin_lock(True)
    try:
        jobs = store.load_jobs()
        if any(j.job_id.startswith(PREFIX) for j in jobs):
            raise RuntimeError('3.8M campaign already exists')
        selected = [j for j in jobs if j.metadata.get('eval_step') == 3768319
                    and j.action != Action.TRAIN_UNTIL_STEP
                    and j.status != JobStatus.SKIPPED]
        ids = {j.job_id: PREFIX + j.job_id for j in selected}
        proposed = []
        for j in selected:
            metadata = rewrite(j.metadata)
            metadata.update(eval_step=3800000, xl_boundary=3800000)
            metadata.pop('skip_reason', None)
            deps = tuple(dict.fromkeys(ids.get(d, TRAIN) for d in j.deps))
            if not deps:
                deps = (TRAIN,)
            if j.action == Action.EXPORT_HF:
                metadata['python_bin'] = str(ROOT / 'scripts/export_dfm15_xl_3800000.sh')
            proposed.append(j.with_updates(job_id=ids[j.job_id], deps=deps,
                metadata=metadata, log_dir=rewrite(j.log_dir),
                status=JobStatus.PENDING, attempt=0))
        assert sum(j.action == Action.EXPORT_HF for j in proposed) == 1
        export_id = next(j.job_id for j in proposed if j.action == Action.EXPORT_HF)
        proposed = [j.with_updates(deps=tuple(dict.fromkeys((*j.deps, export_id))))
                    if j.action in CLIENTS else j for j in proposed]
        assert sum(j.action == Action.AVERAGE for j in proposed) == 9
        assert all(j.metadata.get('no_ema') is not True for j in proposed)
        store.backup(PLAN / f'before-3800k-evals-{int(time.time())}.sqlite')
        store.import_jobs(jobs + proposed, require_admin_lock=True)
        store.export_tsv(PLAN / 'plan.tsv')
        store.backup(PLAN / 'state.backup.sqlite')
        from collections import Counter
        print('Installed', len(proposed), dict(Counter(str(j.action) for j in proposed)))
    finally:
        store.set_admin_lock(False)


def finalize():
    import numpy as np
    from stop_training_at_complete_checkpoint import complete
    directory = ROOT / 'checkpoints/dfm15/XL-from-dfm14-epoch1'
    assert complete(directory, 'step_3800000')
    state = json.loads((directory / 'checkpoint_state_step_3800000.json').read_text())
    assert state['step'] == 3800000
    rows = len(np.load(ROOT / 'data/sampled_dfm15/epoch_0/inst_len.npy', mmap_mode='r'))
    cursor = state['global_row_cursor_in_epoch']
    assert 0 < cursor <= rows
    store = SQLiteJobStore(DB)
    jobs = store.load_jobs()
    parent = next(j for j in jobs if j.action == Action.EXPORT_HF
                  and j.metadata.get('eval_step') == 3768319)
    epoch = float(parent.metadata['eval_epoch']) + cursor / rows
    # The export itself is already claimed but does not log evaluation scores.
    downstream = [j.job_id for j in jobs if j.job_id.startswith(PREFIX)
                  and j.action not in (Action.EXPORT_HF, Action.WAIT_CHECKPOINT)]
    store.finalize_eval_epoch(downstream, epoch)
    (PLAN / 'dfm15-3800k-axis.json').write_text(json.dumps(
        {'step': 3800000, 'epoch': epoch, 'cursor': cursor, 'rows': rows}, indent=2)+'\n')
    print('Finalized fractional eval epoch', epoch)


if __name__ == '__main__':
    finalize() if '--finalize' in sys.argv else install()
