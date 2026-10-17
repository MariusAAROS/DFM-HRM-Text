"""Checkpoint-safe fresh DFM15 cursor, retaining XL optimizer and EMA."""
import json
import os
from pathlib import Path
import subprocess
import sys

from schedule_dfm12_xl_epoch11 import training_arguments
from stop_training_at_complete_checkpoint import complete, preserve

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'checkpoints/dfm14/XL-from-dfm13-step3250000'
OUTPUT = ROOT/'checkpoints/dfm15/XL-from-dfm14-epoch1'
CONTROL = ROOT/'logs/training/dfm15_XL_extension'
START, END = 3768319, 3800000


def reset_state(original):
    if (original['step'] != START or original['carry_policy'] != 'none'
            or original['world_size'] != 8 or original['gradient_accumulation_steps'] != 2):
        raise ValueError('Unexpected DFM14 checkpoint step/carry/world/GAS')
    state = dict(original, epoch=1, batch_in_epoch=0, batch_in_epoch_exact=True,
                 global_row_cursor_in_epoch=0, global_row_start_in_epoch=0,
                 data_path=str(ROOT/'data/sampled_dfm15'))
    state.pop('lr_rewarm', None)
    return state


def command(resume):
    args = training_arguments(dict(start_step=START, end_step=END), [])
    changes = {'data': 'dfm15', 'data.path': str(ROOT/'data/sampled_dfm15'),
               'epochs': '1', 'lr': '1e-5', 'lr_auto': 'true', 'lr_min_ratio': '1',
               'lr_warmup_steps': '0', 'lr_rewarm_steps': '0',
               'lr_rewarm_start_step': 'null', 'lr_decay_start_step': 'null',
               'lr_decay_end_step': 'null', 'arch.bp_warmup_ratio': '0',
               'checkpoint_path': str(OUTPUT), 'resume_checkpoint_path': str(resume),
               'resume_checkpoint_tag': f'step_{START}', 'stop_after_step': str(END)}
    present = {a.split('=', 1)[0] for a in args if '=' in a}
    return [f'{a.split("=",1)[0]}={changes.get(a.split("=",1)[0],a.split("=",1)[1])}'
            if '=' in a else a for a in args] + [f'{k}={v}' for k,v in changes.items() if k not in present]


def main():
    if '--preview' in sys.argv:
        print(json.dumps(command(CONTROL/'resume'), indent=2)); return
    assert (ROOT/'data/sampled_dfm15/local-transfer-verified.json').exists()
    assert (ROOT/'data/sampled_dfm15/full-mixture-verified.json').exists(), 'Full registered DFM15 required, not repair-only base'
    math_receipt = ROOT/'data/sampled_dfm15/math-answer-conditioning-verified.json'
    if not math_receipt.is_file() or json.loads(math_receipt.read_text()).get('version') != 2:
        raise RuntimeError('DFM15 preparation is incomplete; keep the scheduler training hold until math conditioning is verified')
    assert complete(SOURCE, 'epoch_1')
    original = json.loads((SOURCE/'checkpoint_state_epoch_1.json').read_text())
    state = reset_state(original)
    resume = CONTROL/'resume'
    tag = f'step_{START}'
    if not resume.exists():
        stage = CONTROL/'resume.preparing'
        preserve(SOURCE, stage, 'epoch_1')
        (stage/'fsdp2_epoch_1').rename(stage/f'fsdp2_{tag}')
        (stage/f'checkpoint_state_{tag}.json').write_text(json.dumps(state, indent=2)+'\n')
        (stage/'handoff.json').write_text(json.dumps({'original': original, 'resume': state}, indent=2)+'\n')
        stage.rename(resume)
    assert complete(resume, tag)
    assert json.loads((resume/f'checkpoint_state_{tag}.json').read_text()) == state
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['PATH'] = '/home/ucloud/miniforge3/envs/hrm/bin:' + os.environ['PATH']
    subprocess.run(command(resume), cwd=ROOT, check=True)
    assert complete(OUTPUT, f'step_{END}')


if __name__ == '__main__':
    main()
