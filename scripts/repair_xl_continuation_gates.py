"""Rebind pending XL continuation gates to inference, not CPU publication."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval_scheduler'))
from eval_scheduler.database_runner import CLIENTS
from eval_scheduler.model import Action, JobStatus, plan_database
from eval_scheduler.store import SQLiteJobStore


def changes(jobs):
    by_id = {j.job_id: j for j in jobs}
    result = {}
    for target, boundary in [('dfm14-xl-step_3750000-train', 3700000),
                             ('dfm14-xl-epoch_1-train', 3750000)]:
        train = by_id[target]
        if train.status != JobStatus.PENDING:
            raise ValueError(f'{target} must still be pending')
        gate = by_id[target + '-evals-terminal']
        if gate.status == JobStatus.RUNNING or gate.action != Action.TERMINAL_BARRIER:
            raise ValueError('Unexpected gate state/action')
        deps = tuple(j.job_id for j in jobs if j.action in CLIENTS
                     and j.metadata.get('eval_step') == boundary
                     and j.metadata.get('wandb_run_id') == 'dfm8-xl-from-dfm6-dfm7-epoch5-clean-full')
        if not deps:
            raise ValueError(f'No inference found for {boundary}')
        if gate.job_id not in train.deps or train.deps_mode != 'success':
            raise ValueError('Training ancestry/gate contract changed')
        result[gate.job_id] = {'deps': deps, 'deps_mode': 'terminal'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    store = SQLiteJobStore(plan_database(args.plan_dir / 'plan.tsv'))
    edits = changes(store.load_jobs())
    print({key: len(value['deps']) for key, value in edits.items()})
    if not args.apply:
        return
    if store.get_admin_lock():
        raise RuntimeError('Plan already locked')
    store.set_admin_lock(True)
    try:
        store.backup(args.plan_dir / f'before-continuation-gates-{int(time.time())}.sqlite')
        before = store.load_jobs()
        store.update_jobs(changes(before))
        after = store.load_jobs()
        old_status = {j.job_id: j.status for j in before}
        reset = [j.job_id for j in after if j.status == JobStatus.PENDING
                 and old_status[j.job_id] == JobStatus.DONE]
        print('Rearmed:', reset)
        store.export_tsv(args.plan_dir / 'plan.tsv')
        store.backup(args.plan_dir / 'state.backup.sqlite')
    finally:
        store.set_admin_lock(False)


if __name__ == '__main__':
    main()
