"""Add registered DFM15 sources in place without duplicating the inherited token store."""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('inst_start', 'inst_len', 'resp_start', 'resp_len')
TARGET = ROOT/'data/sampled_dfm15'
STAGE = ROOT/'data/sampled_dfm15.building'


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)


def local(path):
    return ROOT/Path(path).relative_to('/work/dfm/HRM-Text')


def append_tokens(path, sources, count):
    """Append payload and rewrite the same-size NPY header; retry is idempotent."""
    with path.open('r+b') as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(f)
        offset = f.tell()
        if fortran or len(shape) != 1:
            raise ValueError('Expected contiguous one-dimensional tokens')
        added = sum(len(a) for a in sources)
        header = io.BytesIO()
        writer = np.lib.format.write_array_header_1_0 if version == (1, 0) else np.lib.format.write_array_header_2_0
        writer(header, dict(descr=np.lib.format.dtype_to_descr(dtype), fortran_order=False, shape=(count+added,)))
        if len(header.getvalue()) != offset:
            raise ValueError('Cannot enlarge NPY header in place')
        if shape[0] not in (count, count+added):
            raise ValueError('Unexpected token store length')
        f.seek(offset+count*dtype.itemsize)
        for a in sources:
            for start in range(0, len(a), 8_000_000):
                chunk = a[start:start+8_000_000]
                if np.any(chunk < 0) or np.any(chunk >= 262144):
                    raise ValueError('Token outside Gemma vocabulary')
                chunk.astype(dtype, copy=False).tofile(f)
        f.truncate(); f.flush(); os.fsync(f.fileno())
        f.seek(0); f.write(header.getvalue()); f.flush(); os.fsync(f.fileno())
    return count+added


def main():
    registry_path = ROOT/'config/dfm15_sources.json'
    registry_bytes = registry_path.read_bytes()
    spec = json.loads(registry_bytes)
    digest = hashlib.sha256(registry_bytes).hexdigest()
    if (TARGET/'full-mixture-verified.json').exists():
        assert json.loads((TARGET/'full-mixture-verified.json').read_text())['registry_sha256'] == digest
        return
    if not STAGE.exists():
        TARGET.rename(STAGE)
    # Never expose a repair-only sample as training-ready while assembling.
    (STAGE/'local-transfer-verified.json').unlink(missing_ok=True)
    files = set()
    for entry in spec['additions']:
        assert entry['repeat'] == 1
        for part in entry['parts']:
            for name in (*[k+'.npy' for k in FIELDS], 'tokens.npy', 'metadata.json'):
                files.add(str(local(part['path']).relative_to(ROOT)/name))
            files.add(str(local(part['selection']).relative_to(ROOT)))
    manifest = STAGE/'addition-files.txt'
    manifest.write_text('\n'.join(sorted(files))+'\n')
    subprocess.run(['rsync', '-aL', '--partial', '--files-from='+str(manifest),
                    '-e', 'ssh -p 6768 -o BatchMode=yes -o ServerAliveInterval=30',
                    'ucloud@ssh.cloud.sdu.dk:/work/dfm/HRM-Text/', str(ROOT)+'/'], check=True)
    base = json.loads((STAGE/'validated.json').read_text())
    token_count = base['stored_tokens']
    pieces = {k: [] for k in FIELDS}
    source_tokens = []
    totals = []
    offset = token_count
    for entry in spec['additions']:
        rows = tokens = rejected = selected_total = 0
        for part in entry['parts']:
            directory = local(part['path'])
            arrays = {k: np.load(directory/(k+'.npy'), mmap_mode='r') for k in FIELDS}
            backing = np.load(directory/'tokens.npy', mmap_mode='r')
            indices = np.load(local(part['selection']))
            assert len(np.unique(indices)) == len(indices)
            assert np.all(indices < len(arrays['inst_len']))
            selected_total += len(indices)
            length = arrays['inst_len'][indices]+arrays['resp_len'][indices]
            # No truncation: omit over-context rows as complete units.
            keep = (length <= 4097) & (arrays['resp_len'][indices] > 0)
            rejected += int((~keep).sum()); indices = indices[keep]
            for prefix in ('inst', 'resp'):
                starts, lengths = arrays[prefix+'_start'][indices], arrays[prefix+'_len'][indices]
                assert np.all(starts <= len(backing)) and np.all(lengths <= len(backing)-starts)
            for k in FIELDS:
                values = arrays[k][indices].astype(np.uint64)
                pieces[k].append(values+(offset if k.endswith('_start') else 0))
            rows += len(indices); tokens += int(length[keep].sum(dtype=np.uint64))
            source_tokens.append(backing); offset += len(backing)
        assert selected_total == entry['selected_rows']
        totals.append(dict(name=entry['name'], repo_id=entry['repo_id'], revision=entry['revision'],
                           repeat=1, rows=rows, tokens=tokens, rejected_length_or_empty=rejected))
    print('Admitted additions', json.dumps(totals), flush=True)
    new = {k: np.concatenate(v) for k,v in pieces.items()}
    stored = append_tokens(STAGE/'tokens.npy', source_tokens, token_count)
    nnew = len(new['inst_start']); added_tokens = sum(t['tokens'] for t in totals)
    reports = []
    for expected in base['epochs']:
        epoch = expected['epoch']; folder = STAGE/f'epoch_{epoch}'
        ready = folder/'full-mixture.json'
        if ready.exists():
            reports.append(json.loads(ready.read_text())); continue
        out = STAGE/f'epoch_{epoch}.building'; out.mkdir(exist_ok=True)
        rng = np.random.default_rng(15000+epoch)
        n = expected['rows']+nnew
        positions = np.sort(rng.choice(n, nnew, replace=False))
        order = rng.permutation(nnew)
        # Preserve inherited sampling/repeats while uniformly interleaving additions.
        for k in FIELDS:
            old = np.load(folder/(k+'.npy'), mmap_mode='r')
            dest = np.lib.format.open_memmap(out/(k+'.npy'), mode='w+', dtype=np.uint64, shape=(n,))
            for start in range(0, n, 1_000_000):
                end = min(n, start+1_000_000)
                indexes = np.arange(start, end, dtype=np.int64)
                ranks = np.searchsorted(positions, indexes)
                is_new = np.zeros(len(indexes), dtype=bool)
                left, right = np.searchsorted(positions, [start, end])
                is_new[positions[left:right]-start] = True
                values = np.empty(len(indexes), dtype=np.uint64)
                values[~is_new] = old[indexes[~is_new]-ranks[~is_new]]
                values[is_new] = new[k][order[left:right]]
                dest[start:end] = values
            dest.flush(); del dest, old
        arrays = {k: np.load(out/(k+'.npy'), mmap_mode='r') for k in FIELDS}
        total = 0
        for start in range(0, n, 1_000_000):
            chunk = {k:a[start:start+1_000_000] for k,a in arrays.items()}
            for prefix in ('inst', 'resp'):
                assert np.all(chunk[prefix+'_start'] <= stored)
                assert np.all(chunk[prefix+'_len'] <= stored-chunk[prefix+'_start'])
                total += int(chunk[prefix+'_len'].sum(dtype=np.uint64))
            assert np.all(chunk['resp_len'] > 0)
            assert np.all(chunk['inst_len']+chunk['resp_len'] <= 4097)
        assert total == expected['tokens']+added_tokens
        report = dict(epoch=epoch, rows=n, tokens=total, added_rows=nnew, added_tokens=added_tokens)
        write_json(out/'full-mixture.json', report)
        np.save(out/'addition_positions.npy', positions)
        del arrays
        shutil.rmtree(folder)
        out.rename(folder)
        reports.append(report)
        print('Verified epoch', report, flush=True)
    metadata = json.loads((STAGE/'metadata.json').read_text())
    metadata['total_length'] = round(sum(r['tokens'] for r in reports)/len(reports))
    metadata['dfm15_registry_sha256'] = digest
    write_json(STAGE/'metadata.json', metadata)
    receipt = dict(registry_sha256=digest, additions=totals, epochs=reports, stored_tokens=stored,
                   base='DFM14 with native reasoning repairs', no_separate_repair_sample=True)
    write_json(STAGE/'full-mixture-verified.json', receipt)
    write_json(STAGE/'local-transfer-verified.json', receipt)
    write_json(STAGE/'ready.json', dict(status='full_registered_dfm15_verified', **receipt))
    write_json(STAGE/'validated.json', dict(epochs=reports, stored_tokens=stored,
               metadata_sha256=hashlib.sha256((STAGE/'metadata.json').read_bytes()).hexdigest()))
    for name in ('token-copy.json', 'source-metadata.json', 'addition-files.txt'):
        (STAGE/name).unlink(missing_ok=True)
    STAGE.rename(TARGET)
    print('FULL DFM15 READY', TARGET, flush=True)


def release_training():
    math_receipt = TARGET/'math-answer-conditioning-verified.json'
    if not math_receipt.is_file() or json.loads(math_receipt.read_text()).get('version') != 2:
        print('Keeping DFM15 training hold: math conditioning is not yet verified', flush=True)
        return
    sys.path.insert(0, str(ROOT/'eval_scheduler'))
    from eval_scheduler.store import SQLiteJobStore
    store = SQLiteJobStore('/tmp/dfm14-production-3383000/jobs.sqlite')
    while store.get_admin_lock():
        time.sleep(5)
    store.set_admin_lock(True)
    try:
        job = next(j for j in store.load_jobs() if j.job_id == 'dfm15-xl-step_3800000-train')
        if not job.metadata.get('dfm15_preparation_hold'):
            return
        assert job.status.value == 'pending'
        metadata = dict(job.metadata, min_gpu_free_mib=178000)
        metadata.pop('dfm15_preparation_hold')
        store.update_jobs({job.job_id: {'metadata': metadata}})
        plan = ROOT/'logs/scheduler/dfm12_XL_epoch11_noidentity'
        store.export_tsv(plan/'plan.tsv')
        store.backup(plan/'state.backup.sqlite')
        print('Released verified DFM15 training hold', flush=True)
    finally:
        store.set_admin_lock(False)


if __name__ == '__main__':
    with (ROOT/'data/.build-dfm15-full.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
        release_training()
