"""One-off verified early CUDA suspension handoff for the 1.1M path repair."""
import json
import argparse
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'eval_scheduler'))
from eval_scheduler.model import Action, Job, JobStatus
from eval_scheduler.store import SQLiteJobStore
from training_suspend.protocol import read_json, matches, validate_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Execute this historical, machine-specific recovery; not safe to replay')
    if not parser.parse_args().execute:
        parser.error('Historical recovery only; --execute is required')
    plan = ROOT/'logs/scheduler/dfm10_XL_epoch9_20260831'
    root = ROOT/'logs/training/dfm14_XXL_wide/cooperative_1m_to_epoch3'
    store = SQLiteJobStore('/tmp/dfm14-xxl-wide-production/jobs.sqlite')
    assert store.get_admin_lock(), 'Dispatch must already be fenced'
    target = read_json(plan/'path-repair-target.json')['requested_step']
    active = Path(read_json(root/'active.json')['control_dir'])
    deadline = time.monotonic()+900
    while True:
        ready = read_json(active/'suspend_ready.json') if (active/'suspend_ready.json').exists() else {}
        if ready.get('step') == target:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Suspension readiness not reached')
        time.sleep(5)
    validate_process(read_json(root/'launcher.json')['identity'])
    # Stop only the scheduler coordinator; supervisor/launcher are independent processes.
    pid = read_json(plan/'runtime.json')['pid']
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
    assert b'-m eval_scheduler run' in cmd
    os.kill(pid, signal.SIGKILL)
    subprocess.run([sys.executable, '-m', 'training_suspend.controller', '--cuda-checkpoint',
        '/home/ucloud/miniforge3/envs/cuda-checkpoint-test/bin/cuda-checkpoint',
        'offload', str(active), '--lease-seconds', '86400'], check=True, timeout=900)
    receipt = read_json(active/f"generation_{ready['generation']:06d}"/'offloaded.json')
    assert matches(receipt, ready)
    resume_id = 'xxlw-dfm14-train-1150000'
    with sqlite3.connect(store.path) as db:
        token = db.execute('select token from jobs where job_id=?', (resume_id,)).fetchone()[0]
    assert store.finish(resume_id, token, {'status': JobStatus.PENDING})
    before = store.load_jobs()
    updates = {}
    retry_ids = []
    gate_deps = []
    config = str(ROOT/'data/dfm14/evaluation/local-config/tasks.yaml')
    for job in before:
        if job.status in (JobStatus.PENDING, JobStatus.FAILED) and str(job.metadata.get('dfm_single_tasks_config', '')).endswith('/config/dfm_evals_dfm14.yaml'):
            updates.setdefault(job.job_id, {})['metadata'] = {**job.metadata, 'dfm_single_tasks_config': config}
        if job.job_id.startswith('xxlw-dfm14-1100000-'):
            if job.action == Action.EVAL_DFM and job.status == JobStatus.FAILED:
                retry_ids.append(job.job_id)
                updates.setdefault(job.job_id, {}).update(status=JobStatus.PENDING, attempt=0)
            if job.status in (JobStatus.PENDING, JobStatus.FAILED) and job.action in (
                Action.EVAL_DFM, Action.EXPORT_DFM_RESULTS, Action.MERGE_DFM, Action.AVERAGE):
                gate_deps.append(job.job_id)
    assert len(retry_ids) == 128
    early = Job(job_id=f'xxlw-dfm14-early-{target}-offloaded', action=Action.TERMINAL_BARRIER,
                family='control', name='verified-early-offload', status=JobStatus.DONE,
                metadata={'step': target, 'generation': ready['generation'], 'offload_receipt': str(active/f"generation_{ready['generation']:06d}"/'offloaded.json')})
    gate = Job(job_id='xxlw-dfm14-1100000-path-repair-terminal', action=Action.TERMINAL_BARRIER,
               family='control', name='repair-exports-merges-averages-terminal',
               deps=tuple(gate_deps), deps_mode='terminal')
    resume = next(j for j in before if j.job_id == resume_id)
    updates[resume_id] = dict(deps=(*resume.deps, early.job_id, gate.job_id),
        metadata={**resume.metadata, 'resume_from_tag': f'step_{target}'})
    definitions = [j.with_updates(**{k:v for k,v in updates.get(j.job_id, {}).items()
                                      if k not in ('status', 'attempt')}) for j in store.load_jobs()]
    store.import_jobs([*definitions, early, gate], preserve_runtime=True, require_admin_lock=True)
    store.update_jobs({i:dict(status=JobStatus.PENDING, attempt=0) for i in retry_ids})
    store.export_tsv(plan/'plan.tsv')
    store.backup(plan/'state.backup.sqlite')
    (plan/'path-repair-installed.json').write_text(json.dumps(dict(step=target,
        retries=retry_ids, gate=gate.job_id, config=config), indent=2))
    (plan/'stop.request').unlink(missing_ok=True)
    store.set_admin_lock(False)
    env = os.environ.copy()
    cuda = '/home/ucloud/miniforge3/envs/hrm/lib/python3.13/site-packages/nvidia/cu13'
    env.update(CUDA_HOME=cuda, PYTHONPATH=str(ROOT)+':'+str(ROOT/'eval_scheduler'),
               PATH='/home/ucloud/miniforge3/envs/hrm/bin:'+cuda+'/bin:'+env['PATH'])
    with (plan/'runner-1100k-path-repair.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-m', 'eval_scheduler', 'run', '--plan-dir', str(plan),
            '--backend', 'sqlite', '--persistent-vllm', '--gpus', '0,1,2,3,4,5,6,7',
            '--max-clients', '64', '--server-concurrency', '1024', '--cpu-cores', '256',
            '--reserve-cores', '8'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('Repair runner', process.pid, 'suspended at', target, flush=True)


if __name__ == '__main__':
    main()
