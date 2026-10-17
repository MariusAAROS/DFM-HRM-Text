"""Transfer and validate DFM15 before publishing it for the XL extension."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '/work/dfm/HRM-Text/data/dfm15/native-reasoning-v2'


def main():
    target = ROOT / 'data/sampled_dfm15'
    if (target / 'local-transfer-verified.json').exists():
        return
    stage = ROOT / 'data/sampled_dfm15.transferring'
    stage.mkdir(exist_ok=True)
    base = ['rsync', '-aL', '--partial', '--info=progress2', '-e',
            'ssh -p 6768 -o BatchMode=yes -o ServerAliveInterval=30']
    subprocess.run([*base, f'ucloud@ssh.cloud.sdu.dk:{SOURCE}/sampled_dfm15/', str(stage)+'/'], check=True)
    subprocess.run([*base, f'ucloud@ssh.cloud.sdu.dk:{SOURCE}/chat_template.jinja', str(stage)+'/'], check=True)
    original = (stage/'metadata.json').read_bytes()
    validated = json.loads((stage/'validated.json').read_text())
    assert hashlib.sha256(original).hexdigest() == validated['metadata_sha256']
    tokenizer = ROOT.parent/'brainsurgery/models/gemma4_31b/tokenizer.json'
    assert hashlib.sha256(tokenizer.read_bytes()).hexdigest() == '12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6'
    tokens = np.load(stage/'tokens.npy', mmap_mode='r')
    assert len(tokens) == validated['stored_tokens']
    for expected in validated['epochs']:
        arrays = {k: np.load(stage/f"epoch_{expected['epoch']}"/(k+'.npy'), mmap_mode='r')
                  for k in ('inst_start', 'inst_len', 'resp_start', 'resp_len')}
        assert {len(a) for a in arrays.values()} == {expected['rows']}
        total = 0
        for offset in range(0, expected['rows'], 1_000_000):
            chunk = {k: a[offset:offset+1_000_000] for k, a in arrays.items()}
            for prefix in ('inst', 'resp'):
                start, size = chunk[prefix+'_start'], chunk[prefix+'_len']
                assert np.all(start <= len(tokens)) and np.all(size <= len(tokens)-start)
                total += int(size.sum(dtype=np.uint64))
            assert np.all(chunk['resp_len'] > 0)
            assert np.all(chunk['inst_len']+chunk['resp_len'] <= 4097)
        assert total == expected['tokens']
        print('Validated', expected, flush=True)
    metadata = json.loads(original)
    metadata['tokenizer_info']['chat_template_path'] = str(target/'chat_template.jinja')
    (stage/'source-metadata.json').write_bytes(original)
    (stage/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (stage/'local-transfer-verified.json').write_text(json.dumps({
        'source': SOURCE, 'validated': validated,
        'transport': 'rsync transfer checksum; full index bounds and totals checked',
    }, indent=2)+'\n')
    os.rename(stage, target)
    print('READY', target, flush=True)


if __name__ == '__main__':
    with (ROOT/'data/.transfer-dfm15.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
