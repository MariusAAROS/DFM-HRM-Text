"""Append the DFM15 extension only after its transferred sample is verified."""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'eval_scheduler'))
from eval_scheduler.database_runner import CLIENTS
from eval_scheduler.model import Action, Job, plan_database
from eval_scheduler.store import SQLiteJobStore
from run_dfm15_xl_extension import CONTROL, OUTPUT, START, END

PLAN = ROOT/'logs/scheduler/dfm12_XL_epoch11_noidentity'
RUN = 'dfm8-xl-from-dfm6-dfm7-epoch5-clean-full'


def additions(jobs):
    deps = tuple(j.job_id for j in jobs if j.action in CLIENTS
                 and j.metadata.get('eval_step') == START and j.metadata.get('wandb_run_id') == RUN)
    assert deps, 'No final DFM14 inference rows'
    assert any(j.job_id == 'dfm14-xl-epoch_1-train' for j in jobs)
    gate = Job(job_id='dfm15-xl-final-dfm14-inference-terminal',
               action=Action.TERMINAL_BARRIER, family='control', name='dfm14-final-inference',
               deps=deps, deps_mode='terminal', log_dir=str(CONTROL/'gate'))
    train = Job(job_id='dfm15-xl-step_3800000-train', action=Action.TRAIN_UNTIL_STEP,
                family='training', name='step_3800000', deps=('dfm14-xl-epoch_1-train', gate.job_id),
                gpu_policy='all', gpu_count=8, max_retries=0, log_dir=str(CONTROL/'training'),
                metadata={'command': ['/home/ucloud/miniforge3/envs/hrm/bin/python',
                            str(ROOT/'scripts/run_dfm15_xl_extension.py')],
                          'workdir': str(ROOT), 'ckpt_path': str(OUTPUT),
                          'ckpt_tag': 'step_3800000', 'stop_after_step': END,
                          'checkpoint_carry_ranks': 8, 'min_gpu_free_mib': 178000,
                          'eval_step': END, 'wandb_run_id': RUN})
    return [gate, train]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    store = SQLiteJobStore(plan_database(PLAN/'plan.tsv'))
    proposed = additions(store.load_jobs())
    if args.preview:
        print([(j.job_id, len(j.deps)) for j in proposed]); return
    ready = ROOT/'data/sampled_dfm15/local-transfer-verified.json'
    while not ready.exists():
        if not args.wait:
            raise RuntimeError('DFM15 transfer/validation not complete')
        # Recover a failed transfer owner without duplicate writers.
        owners = [p for p in Path('/proc').glob('[0-9]*/cmdline')
                  if p.exists()]
        active = False
        for p in owners:
            try:
                parts = p.read_bytes().split(b'\0')
                active |= any(x.endswith(b'/prepare_dfm15_extension.py')
                              or x == b'scripts/prepare_dfm15_extension.py' for x in parts)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        if not active:
            with (CONTROL/'prepare.log').open('ab') as log:
                subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/prepare_dfm15_extension.py')],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print('Waiting for verified DFM15', flush=True); time.sleep(30)
    while store.get_admin_lock():
        time.sleep(5)
    store.set_admin_lock(True)
    try:
        jobs = store.load_jobs()
        if any(j.job_id == proposed[-1].job_id for j in jobs):
            return
        store.backup(CONTROL/f'before-install-{int(time.time())}.sqlite')
        store.import_jobs(jobs+additions(jobs), require_admin_lock=True)
        store.export_tsv(PLAN/'plan.tsv')
        store.backup(PLAN/'state.backup.sqlite')
        (CONTROL/'installed.json').write_text(json.dumps({'step_start': START, 'step_end': END,
            'base_lr': 1e-5, 'run': RUN, 'data_ready': str(ready)}, indent=2)+'\n')
        print('Installed DFM15 continuation through 3800000', flush=True)
    finally:
        store.set_admin_lock(False)


if __name__ == '__main__':
    main()
