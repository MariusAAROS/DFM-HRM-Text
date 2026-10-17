"""Deferred evaluation-summary repair; never appends history or starts training."""
import json
import os
from pathlib import Path
import time
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
RUN = 'dfm8-xl-from-dfm6-dfm7-epoch5-clean-full'
PATH = 'peter-sk-sdu/DFM5/' + RUN
OUT = ROOT / 'logs/repairs/humaneval-dfm14-final-20261017'


def evaluations_pending():
    with sqlite3.connect('file:/tmp/dfm14-production-3383000/jobs.sqlite?mode=ro', uri=True) as db:
        return db.execute("SELECT count(*) FROM jobs WHERE job_id LIKE 'dfm15-xl-3800000-eval-%' AND status IN ('pending','running')").fetchone()[0]


def writers():
    found = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if int(p.parent.name) != os.getpid() and RUN in args:
            found.append(int(p.parent.name))
    return found


def main():
    from stop_training_at_complete_checkpoint import complete
    import wandb

    checkpoint = ROOT / 'checkpoints/dfm15/XL-from-dfm14-epoch1'
    while not complete(checkpoint, 'step_3800000') or writers() or evaluations_pending():
        print(time.strftime('%FT%T'), 'Waiting for completed 3.8M and writer exit', writers(), flush=True)
        time.sleep(120)
    # Allow final SDK flush, then require quiet remote history before mutation.
    while True:
        run = wandb.Api().run(PATH)
        last = run.lastHistoryStep
        time.sleep(120)
        run = wandb.Api().run(PATH)
        if writers() or evaluations_pending() or run.lastHistoryStep != last:
            continue
        updates = json.loads((OUT / 'average-summary-repair.json').read_text())
        updates['dfm_eval/humaneval/verify_sanitized/accuracy'] = 126 / 164
        latest = {}
        for row in run.scan_history(min_step=max(0, last - 20000), page_size=1000):
            for key, step in row.items():
                if not key.endswith('/train_step') or not isinstance(step, (int, float)):
                    continue
                prefix = key.rsplit('/', 1)[0]
                if prefix not in ('eval', 'dfm_eval', 'euroeval') and 'avg' not in prefix:
                    continue
                for name, value in row.items():
                    if name.startswith(prefix + '/') and value is not None:
                        rank = (step, row.get('_step', 0))
                        if name not in latest or rank >= latest[name][0]:
                            latest[name] = (rank, value)
        updates.update({k: v for k, (_, v) in latest.items()})
        if writers() or evaluations_pending() or wandb.Api().run(PATH).lastHistoryStep != last:
            continue
        run.summary.update(updates)
        time.sleep(60)
        actual = dict(wandb.Api().run(PATH).summary)
        missing = {k: {'expected': v, 'actual': actual.get(k)}
                   for k, v in updates.items() if actual.get(k) != v}
        receipt = {'run': PATH, 'history_step': last, 'updates': updates,
                   'mismatches': missing, 'verified': not missing}
        (OUT / 'deferred-summary-rebuild.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if not missing:
            print('VERIFIED evaluation summary rebuilt', len(updates), flush=True)
            return
        print('Verification failed; retrying after quiet window', len(missing), flush=True)
        time.sleep(300)


if __name__ == '__main__':
    main()
