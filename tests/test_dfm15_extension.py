import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_dfm15_xl_extension import command, reset_state, START, END


def test_reset_only_cursor_not_global_step():
    original = dict(step=START, epoch=1, batch_in_epoch=518319,
                    global_row_cursor_in_epoch=406830651, world_size=8,
                    gradient_accumulation_steps=2, carry_policy='none', sentinel='preserved')
    result = reset_state(original)
    assert result['step'] == START and result['epoch'] == 1
    assert result['batch_in_epoch'] == result['global_row_cursor_in_epoch'] == 0
    assert result['sentinel'] == 'preserved'
    assert original['batch_in_epoch'] == 518319


def test_command_retains_run_and_optimizer_policy():
    args = dict(a.split('=', 1) for a in command(Path('/resume')) if '=' in a)
    assert args['data'] == 'dfm15'
    assert args['wandb_run_id'] == 'dfm8-xl-from-dfm6-dfm7-epoch5-clean-full'
    assert args['reset_ema_on_resume'] == 'false'
    assert args['gradient_accumulation_steps'] == '2'
    assert args['arch.bp_max_steps'] == '8'
    assert args['lr'] == '1e-5' and args['lr_auto'] == 'true'
    assert args['lr_min_ratio'] == '1' and args['lr_rewarm_steps'] == '0'
    assert args['stop_after_step'] == args['training_total_steps'] == str(END)
    assert END-START == 31681
