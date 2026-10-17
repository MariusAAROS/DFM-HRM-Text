"""Enable suspension only for pending DFM14 XL segments after a chosen boundary."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval_scheduler'))

from eval_scheduler.model import Action, JobStatus, plan_database
from eval_scheduler.store import SQLiteJobStore
from scripts import schedule_dfm14_xl_handoff as handoff
from training_suspend.protocol import read_json
from training_suspend.supervisor import overrides


def prepare(jobs, plan, control, settings, after_step):
    training = sorted((j for j in jobs if j.job_id.startswith('dfm14-xl-')
                       and j.action == Action.TRAIN_UNTIL_STEP
                       and int(j.metadata['stop_after_step']) > after_step),
                      key=lambda j: int(j.metadata['stop_after_step']))
    if len(training) < 2 or any(j.status != JobStatus.PENDING for j in training):
        raise RuntimeError('Expected at least two pending continuation rows')
    if any(j.metadata.get('cooperative_training') for j in training):
        raise RuntimeError('Suspension already configured; inspect before changing it')
    if len({str(j.metadata['command']) for j in training}) != 1:
        raise RuntimeError('Training commands differ; do not flatten the schedule')
    run = handoff.read_prepared(handoff.CONTROL)
    if training[-1].metadata['stop_after_step'] != run['end_step'] + 1:
        raise RuntimeError('Final segment must retain epoch-completion semantics')
    config = read_json(settings)
    for key in ('shim_library', 'nccl_library', 'redis_server', 'cuda_checkpoint'):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    boundaries = [int(j.metadata['stop_after_step']) for j in training[:-1]]
    command = ['OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', *overrides(
        handoff.training_arguments(run, []), {
            'stop_after_step': training[-1].metadata['stop_after_step'],
            'suspend_after_steps': str(boundaries).replace(' ', ''),
            'suspend_poll_interval_steps': 10, 'suspend_wait_poll_seconds': 2,
        })]
    by_id = {j.job_id: j for j in jobs}
    changes = {}
    for job in training:
        target = job.metadata['stop_after_step']
        gate = by_id[job.job_id + '-evals-terminal']
        if gate.deps_mode != 'terminal' or not gate.deps:
            raise RuntimeError('Training must retain its terminal inference gate')
        co = dict(control_root=str(control), command=command,
                  python=sys.executable, launcher_settings=str(settings),
                  cuda_checkpoint=config['cuda_checkpoint'], final_boundary=job == training[-1],
                  finalize_command=[sys.executable, str(handoff.SCRIPT), 'finalize',
                                    '--plan-dir', str(plan), '--control', str(handoff.CONTROL),
                                    f'stop_after_step={target}'])
        changes[job.job_id] = {'metadata': {**job.metadata, 'cooperative_training': co}}
    return changes, boundaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--after-step', type=int, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = args.plan_dir.resolve()
    database = plan_database(plan / 'plan.tsv')
    if not database:
        raise RuntimeError('Requires the SQLite production plan')
    store = SQLiteJobStore(database)
    def calculate():
        return prepare(store.load_jobs(), plan, args.control_root.resolve(),
                       args.settings.resolve(), args.after_step)
    changes, boundaries = calculate()
    print(f'{len(changes)} pending training rows; suspensions {boundaries}')
    if not args.apply:
        return
    if store.get_admin_lock():
        raise RuntimeError('Another administrator holds the plan lock')
    store.set_admin_lock(True)
    try:
        changes, _ = calculate()
        backup = plan / f'before-xl-suspension-after-{args.after_step}.sqlite'
        if backup.exists():
            raise RuntimeError('Backup already exists; inspect before applying again')
        store.backup(backup)
        store.update_jobs(changes)
        store.export_tsv(plan / 'plan.tsv')
        store.backup(plan / 'state.backup.sqlite')
    finally:
        store.set_admin_lock(False)
    print('Pending rows configured; running training and evaluation rows untouched')


if __name__ == '__main__':
    main()
