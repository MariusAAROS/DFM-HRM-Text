"""Archive resume/evaluation artifacts without changing either host's campaigns."""
import argparse
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'logs/transfers/xl-20261015'
REMOTE = 'ucloud@ssh.cloud.sdu.dk'
DEST = '/work/dfm/HRM-Text/incoming/mimir-xl-20261015'
SSH = ['ssh', '-p', '6781', '-o', 'HostKeyAlias=[ssh.cloud.sdu.dk]:6768',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6']
DB = Path('/tmp/dfm14-production-3383000/jobs.sqlite')
END = 3768319


def run(args, **kwargs):
    print(time.strftime('%FT%T'), ' '.join(map(str, args)), flush=True)
    return subprocess.run(args, check=True, **kwargs)


def final_ready():
    from stop_training_at_complete_checkpoint import complete
    extension = ROOT / 'checkpoints/dfm15/XL-from-dfm14-epoch1/checkpoint_state_step_3800000.json'
    if not extension.exists() or json.loads(extension.read_text()).get('step') != 3800000:
        return False
    if not complete(extension.parent, 'step_3800000'):
        return False
    state = ROOT / 'checkpoints/dfm14/XL-from-dfm13-step3250000/checkpoint_state_epoch_1.json'
    if not state.exists() or json.loads(state.read_text()).get('step') != END:
        return False
    with sqlite3.connect(f'file:{DB}?mode=ro', uri=True) as db:
        rows = db.execute('select status,definition from jobs').fetchall()
    final = []
    for status, definition in rows:
        job = json.loads(definition)
        meta = json.loads(job.get('metadata_json') or '{}')
        if meta.get('eval_step') == END and meta.get('wandb_run_id') == 'dfm8-xl-from-dfm6-dfm7-epoch5-clean-full':
            final.append(status)
    # Exhausted jobs must be archived with their evidence, not block backups.
    return bool(final) and all(s in ('done', 'skipped', 'failed') for s in final)


def transfer(final=False, resume_only=False):
    label = 'final' if final else 'initial'
    snap = CONTROL / label
    snap.mkdir(parents=True, exist_ok=True)
    # SQLite backup includes committed WAL content, unlike copying live DB files.
    with closing(sqlite3.connect(f'file:{DB}?mode=ro', uri=True)) as src:
        with closing(sqlite3.connect(snap / 'scheduler-backup.sqlite')) as dst:
            src.backup(dst)
            dst.execute('pragma wal_checkpoint(truncate)')
            dst.execute('pragma journal_mode=delete')
    (snap / 'git-head.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True))
    (snap / 'submodules.txt').write_text(subprocess.check_output(['git', 'submodule', 'status', '--recursive'], cwd=ROOT, text=True))
    with (snap / 'working-tree.patch').open('wb') as f:
        run(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT, stdout=f)
    bundle = snap / 'repository.bundle'
    if bundle.exists():
        bundle.unlink()
    run([*SSH, REMOTE, f'mkdir -p {DEST}/transfer-state/{label}'])
    base = ['rsync', '-aL', '--partial', '--partial-dir=.rsync-partial', '--stats',
            '-e', ' '.join(SSH)]
    # Small provenance and the scheduler snapshot go first.
    run([*base, str(snap) + '/', f'{REMOTE}:{DEST}/transfer-state/{label}/'])
    source_manifest = snap / 'source.files'
    source_manifest.write_bytes(subprocess.check_output(
        ['git', 'ls-files', '--recurse-submodules', '-z'], cwd=ROOT))
    run([*base, '--from0', '--files-from=' + str(source_manifest),
         str(ROOT) + '/', f'{REMOTE}:{DEST}/'])
    paths = ['data/dfm11_tokenizer', 'data/sampled_dfm15',
             'data/sampled_dfm14', 'data/dfm15', 'data/dfm14',
             'config', 'wiki', 'docs', 'scripts', 'utils',
             'evaluation', 'eval_scheduler']
    if not resume_only:
        paths += ['exports', 'data/eval', 'data/evaluation', 'logs/training',
                  'logs/eval', 'logs/dfm_evals', 'logs/euroeval',
                  'logs/scheduler', 'logs/reports']
    # Only checkpoint tags with published state markers; never rotating ephemerals.
    selected = set()
    for state in (ROOT / 'checkpoints').rglob('checkpoint_state_*.json'):
        tag = state.stem.removeprefix('checkpoint_state_')
        if 'ephemeral' in tag or not tag.startswith(('step_', 'epoch_')):
            continue
        payloads = list(state.parent.glob('*' + tag + '*'))
        valid = [p for p in payloads if p.is_dir() and (p / '.metadata').is_file()]
        valid += [p for p in payloads if p.is_file() and p.suffix in ('.pt', '.pth', '.safetensors')]
        if not valid:
            continue
        selected.add(str(state.relative_to(ROOT)))
        for payload in valid:
            for p in (payload.rglob('*') if payload.is_dir() else [payload]):
                if p.is_file():
                    selected.add(str(p.relative_to(ROOT)))
        for name in ('all_config.yaml', 'train_metadata.yaml', 'hrm_nocarry_bp_warmup.py'):
            p = state.parent / name
            if p.is_file():
                selected.add(str(p.relative_to(ROOT)))
    manifest = snap / 'checkpoints.files'
    manifest.write_bytes(b''.join(os.fsencode(p) + b'\0' for p in sorted(selected)))
    def is_xl(path):
        return any(part.startswith('XL') and not part.startswith('XXL') for part in Path(path).parts)

    current = sorted(p for p in selected if p.startswith(('checkpoints/dfm14/XL-', 'checkpoints/dfm15/XL-')))
    other_xl = sorted(p for p in selected if is_xl(p) and p not in current)
    historical = sorted(selected - set(current) - set(other_xl))

    def copy_checkpoints(name, files):
        if not files:
            return
        print(name, 'files', len(files), 'bytes', sum((ROOT / p).stat().st_size for p in files), flush=True)
        probe = ('import json,os,sys; result={}\n'
                 'for p in json.load(sys.stdin):\n'
                 ' try:\n'
                 f'  s=os.stat(os.path.join({DEST!r},p)); result[p]=[s.st_size,int(s.st_mtime)]\n'
                 ' except FileNotFoundError: pass\n'
                 'print(json.dumps(result))')
        remote = json.loads(subprocess.check_output(
            [*SSH, REMOTE, 'python3 -c ' + shlex.quote(probe)],
            input=json.dumps(files), text=True))
        sizes = {}
        for p in files:
            s = (ROOT / p).stat()
            sizes[p] = 0 if remote.get(p) == [s.st_size, int(s.st_mtime)] else s.st_size
        # Balance outstanding bytes, not already-copied files. Smaller work
        # units let free streams pick up later work rather than idle at tails.
        batches, loads = [[] for _ in range(32)], [0] * 32
        for p in sorted(files, key=lambda p: sizes[p], reverse=True):
            index = min(range(32), key=lambda i: (loads[i], len(batches[i])))
            batches[index].append(p)
            loads[index] += sizes[p]
        print(name, 'remaining bytes (rsync quick check)', sum(loads), flush=True)
        # Disjoint file ownership: parallel streams never write the same payload.
        manifests = []
        for index, batch in enumerate(batches):
            if batch:
                part = snap / f'{name}-{index}.files'
                part.write_bytes(b''.join(os.fsencode(p) + b'\0' for p in batch))
                manifests.append(part)
        def copy(part):
            run([*base, '--from0', '--files-from=' + str(part), str(ROOT) + '/', f'{REMOTE}:{DEST}/'])
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy, manifests))
        (snap / f'{name}.complete').write_text(time.strftime('%FT%T') + '\n')

    copy_checkpoints('current-xl', current)
    copy_checkpoints('other-xl', other_xl)
    for path in paths:
        if not (ROOT / path).exists():
            continue
        # Preserve historical log links, including dangling links, without
        # traversing obsolete compiler caches or outside archive roots.
        options = [arg if arg != '-aL' else '-a' for arg in base] if path.startswith('logs/') else base
        # Reuse identical files already on the receiving machine. Normal rsync
        # replacement is atomic; never use --inplace with these shared inodes.
        if path.startswith('data/'):
            options = [*options, '--link-dest=/work/dfm/HRM-Text']
        run([*options, '--relative', '--exclude=*.sqlite*', '--exclude=*.db*',
             '--exclude=*.lock', '--exclude=*.sock', '--exclude=__pycache__',
             '--exclude=.git', '--exclude=*.tmp', '--exclude=wandb',
             '--exclude=*.vllm.cache', '--exclude=.cache', '--exclude=triton',
             '--exclude=torchinductor*', './' + path, f'{REMOTE}:{DEST}/'], cwd=ROOT)
    tokenizer = ROOT.parent / 'brainsurgery/models/gemma4_31b/tokenizer.json'
    if tokenizer.exists():
        run([*base, str(tokenizer), f'{REMOTE}:{DEST}/transfer-state/tokenizer.json'])
    if not resume_only:
        copy_checkpoints('historical-checkpoints', historical)
    run(['git', 'bundle', 'create', str(bundle), '--all'], cwd=ROOT)
    receipt = {'completed_at': time.strftime('%FT%T'), 'phase': label, 'destination': DEST,
               'resume_only': resume_only,
               'checkpoint_files': len(current) + len(other_xl) if resume_only else len(selected), 'verification': 'rsync transfer checksums; no independent full archive hash',
               'source_root': str(ROOT), 'epoch_start': 3250000, 'epoch_end': END,
               'dfm15_extension_end': 3800000,
               'note': 'Archive only. Receiver must remap absolute paths and construct its own XL plan.'}
    (snap / ('resume-receipt.json' if resume_only else 'receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    run([*base, str(snap) + '/', f'{REMOTE}:{DEST}/transfer-state/{label}/'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--watch-final', action='store_true')
    args = parser.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / 'transfer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def retry_transfer(final, resume_only=False):
            while True:
                try:
                    transfer(final, resume_only=resume_only)
                    return
                except (OSError, subprocess.CalledProcessError) as exc:
                    print(time.strftime('%FT%T'), 'Transfer interrupted; retry in 300s:', exc, flush=True)
                    time.sleep(300)
        retry_transfer(args.final, resume_only=args.watch_final)
        if args.watch_final:
            while not final_ready():
                print(time.strftime('%FT%T'), 'Waiting for final checkpoint and terminal evaluations', flush=True)
                time.sleep(300)
            retry_transfer(True, resume_only=True)
            # Bulk logs must not delay the final resumable checkpoint/data copy.
            retry_transfer(True)


if __name__ == '__main__':
    main()
