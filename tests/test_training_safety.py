import hashlib
import json

import pytest

from utils.training_safety import check_training_pins
from scripts import schedule_dfm14_xl_handoff as h


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_code_edits_warn_and_record_without_repinning(tmp_path, capsys):
    code = tmp_path / 'pretrain.py'
    code.write_text('old')
    pins = {str(code): digest(code)}
    original = pins.copy()
    code.write_text('new')
    audit = tmp_path / 'audit.jsonl'
    changes = check_training_pins(pins, audit_path=audit)
    assert len(changes) == 1 and pins == original
    assert json.loads(audit.read_text())['changes'] == changes
    assert 'continuing' in capsys.readouterr().err


@pytest.mark.parametrize('name', ['metadata.json', 'data.yaml', 'indices.npy'])
def test_changed_or_missing_inputs_still_block(tmp_path, name):
    path = tmp_path / name
    path.write_text('original')
    pins = {str(path): digest(path)}
    path.write_text('changed')
    with pytest.raises(ValueError, match='input/configuration'):
        check_training_pins(pins)
    path.unlink()
    with pytest.raises(ValueError, match='input/configuration'):
        check_training_pins(pins)


def test_audit_write_failure_does_not_stop_training(tmp_path, capsys):
    code = tmp_path / 'launcher.sh'
    code.write_text('new')
    check_training_pins({str(code): 'old'}, audit_path=tmp_path)
    assert 'Cannot record' in capsys.readouterr().err


def test_prepared_budget_change_still_blocks(tmp_path):
    (tmp_path / 'prepared.json').write_text('{}')
    (tmp_path / 'installed.json').write_text(json.dumps({'prepared_sha256': 'wrong', 'pins': {}}))
    with pytest.raises(ValueError, match='budget/data contract'):
        h.validate_contract(tmp_path)


def test_code_change_reaches_data_validation(tmp_path, monkeypatch):
    prepared = tmp_path / 'prepared.json'
    prepared.write_text('{}')
    code = tmp_path / 'launcher.py'
    code.write_text('changed')
    (tmp_path / 'installed.json').write_text(json.dumps({
        'prepared_sha256': digest(prepared), 'pins': {str(code): 'old'}}))
    monkeypatch.setattr(h, 'read_prepared', lambda _: {'verified': True})
    assert h.validate_contract(tmp_path) == {'verified': True}


def command_fixture():
    run = {'dataset': '/data', 'end_step': 3768319, 'checkpoint_path': '/checkpoints',
           'wandb_run_id': 'original', 'wandb_project': 'DFM5'}
    command = ['torchrun', 'pretrain.py', 'data=dfm14', 'data.path=/data',
               'arch/size@arch=XL', 'global_batch_size=262144',
               'gradient_accumulation_steps=2', 'epochs=1', 'training_total_steps=3768319',
               'checkpoint_path=/checkpoints', 'wandb_run_id=original', 'project_name=DFM5',
               'lr=3e-4', 'lr_auto=true']
    return run, command, {'approved_lr_schedule': {'lr': 0.0003, 'lr_auto': True}}


def test_equivalent_lr_not_code_pin_block():
    run, command, metadata = command_fixture()
    h.validate_command(command, run, metadata)


@pytest.mark.parametrize('override', ['lr=1e-3', 'gradient_accumulation_steps=4',
                                     'data.path=/wrong', 'wandb_run_id=wrong',
                                     'training_total_steps=4000000', 'arch/size@arch=XXL'])
def test_actual_command_drift_blocks(override):
    run, command, metadata = command_fixture()
    with pytest.raises(ValueError, match='command contract mismatch'):
        h.validate_command(command + [override], run, metadata)
