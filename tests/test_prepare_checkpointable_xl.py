from pathlib import Path

import pytest

from scripts import prepare_checkpointable_xl as p
from eval_scheduler.model import Action, Job, JobStatus


def setup(monkeypatch, tmp_path):
    settings = tmp_path / 'settings.json'
    settings.write_text('{}')
    monkeypatch.setattr(p, 'read_json', lambda _: dict.fromkeys(
        ('shim_library', 'nccl_library', 'redis_server', 'cuda_checkpoint'), str(settings)))
    monkeypatch.setattr(p.handoff, 'read_prepared', lambda _: {'end_step': 3768319})
    monkeypatch.setattr(p.handoff, 'training_arguments', lambda *_: [
        'torchrun', 'pretrain.py', 'lr=3e-4', 'lr_auto=true',
        'lr_decay_start_step=3500000', 'lr_decay_end_step=3750000'])
    jobs = []
    for step in (3650000, 3700000, 3750000, 3768320):
        key = f'dfm14-xl-{step}'
        jobs.append(Job(key, Action.TRAIN_UNTIL_STEP, 'training', 'segment',
                        status=JobStatus.RUNNING if step == 3650000 else JobStatus.PENDING,
                        metadata={'stop_after_step': step, 'command': 'unchanged',
                                  'resume_from_tag': f'step_{step - 50000}'}))
        jobs.append(Job(key + '-evals-terminal', Action.TERMINAL_BARRIER,
                        'gate', 'gate', deps=('eval',), deps_mode='terminal'))
    return jobs, settings


def test_only_future_rows_and_preserve_lr(monkeypatch, tmp_path):
    jobs, settings = setup(monkeypatch, tmp_path)
    changes, boundaries = p.prepare(jobs, tmp_path, tmp_path / 'control', settings, 3650000)
    assert len(changes) == 3
    assert jobs[0].job_id not in changes
    assert boundaries == [3700000, 3750000]
    for key, change in changes.items():
        original = next(j for j in jobs if j.job_id == key)
        metadata = change['metadata']
        assert {k: metadata[k] for k in original.metadata} == original.metadata
        co = metadata['cooperative_training']
        assert co['final_boundary'] == (original.metadata['stop_after_step'] == 3768320)
        assert 'lr_auto=true' in co['command']
        assert 'lr_decay_start_step=3500000' in co['command']
        assert 'stop_after_step=3768320' in co['command']
        assert 'suspend_after_steps=[3700000,3750000]' in co['command']
        assert 'finalize' in co['finalize_command']


def test_reject_active_continuation(monkeypatch, tmp_path):
    jobs, settings = setup(monkeypatch, tmp_path)
    jobs[2] = jobs[2].with_updates(status=JobStatus.RUNNING)
    with pytest.raises(RuntimeError, match='pending'):
        p.prepare(jobs, tmp_path, tmp_path / 'control', settings, 3650000)


def test_reject_success_only_gate(monkeypatch, tmp_path):
    jobs, settings = setup(monkeypatch, tmp_path)
    jobs[3] = jobs[3].with_updates(deps_mode='success')
    with pytest.raises(RuntimeError, match='terminal inference gate'):
        p.prepare(jobs, tmp_path, tmp_path / 'control', settings, 3650000)
