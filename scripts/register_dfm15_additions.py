"""Register staged additions and verify exact overlap within the new conversations."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from dfm12.io import load, lock, write_json


def fingerprint(messages):
    normalized = [{k: unicodedata.normalize('NFC', v).strip()
                   for k, v in m.items() if k in ('role', 'content', 'reasoning_content')}
                  for m in messages]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False,
                                    sort_keys=True).encode()).hexdigest()


def main():
    root = Path('data/dfm15')
    with lock(root/'additions-registration.lock'):
        source = hf_hub_download('Setur/fo-instruct', 'data/train.jsonl', repo_type='dataset',
                                revision='55a97b043e1a3f2778492c72bf124f4091341402')
        old = Counter()
        for line in open(source):
            row = json.loads(line)
            old[fingerprint([dict(role='user', content=row['instruction']),
                             dict(role='assistant', content=row['output'])])] += 1
        composite = pq.read_table(root/'dyna-instruct/faroese-dyna-instruct/download/data/fo-instruct/fo-instruct.parquet')
        new = Counter(fingerprint(r['messages']) for r in composite.to_pylist())
        if old != new:
            raise ValueError('FO-Instruct changed; blanket constituent exclusion is unsafe')
        additions = []
        seen = set(old)
        evidence = dict(setur_exact_multiset_match=True, excluded_setur_rows=sum(new.values()),
                        scope='New additions and pinned inherited Setur; not a full inherited row scan',
                        additions=[])
        for manifest in [root/'dyna-instruct/additions.json', root/'small-additions/additions.json']:
            for entry in load(manifest)['additions']:
                folder = Path(entry['tokenized_path']).parent
                repeated = rows = tokens = long = 0
                parts = []
                for source_file in sorted((folder/'input').glob('*.jsonl')):
                    part = folder/'tokenized'/source_file.name
                    lengths = np.load(part/'inst_len.npy', mmap_mode='r')+np.load(part/'resp_len.npy', mmap_mode='r')
                    keep = np.ones(len(lengths), dtype=np.bool_)
                    count = 0
                    for i, line in enumerate(source_file.open()):
                        count += 1
                        key = fingerprint(json.loads(line)['messages'])
                        if key in seen:
                            keep[i] = False
                            repeated += 1
                        seen.add(key)
                    if count != len(lengths):
                        raise ValueError('Source/tokenized row mismatch')
                    selection = folder/'selection'/f'{source_file.stem}.npy'
                    selection.parent.mkdir(exist_ok=True)
                    np.save(selection, np.flatnonzero(keep).astype(np.uint64))
                    rows += int(keep.sum())
                    tokens += int(lengths[keep].sum())
                    long += int((lengths[keep] > 4096).sum())
                    parts.append(dict(path=str(part.resolve()), selection=str(selection.resolve())))
                additions.append(dict(**entry, selected_rows=rows, selected_tokens=tokens,
                    duplicate_rows_excluded=repeated, selected_sequences_over_4096=long, parts=parts))
                evidence['additions'].append(dict(name=entry['name'], rows=rows, tokens=tokens,
                                                  duplicate_rows_excluded=repeated))
        write_json(root/'overlap-report.json', evidence)
        write_json('config/dfm15_sources.json', dict(inherits='dfm14',
            base='data/dfm15/native-reasoning-v2/sampled_dfm15', additions=additions,
            status='tokenized_staging_not_resampled', selection_policy='Use each part selection index; do not sample unfiltered backing rows',
            full_cross_corpus_row_overlap_checked=False))
        print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
