"""Prepare the authorized Faroese/Icelandic additions without touching live samples."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

from dfm12.io import load, lock, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be between 1 and 16')
    root = Path('data/dfm15/dyna-instruct').resolve()
    root.mkdir(parents=True, exist_ok=True)
    with lock(root/'.lock'):
        metadata = load('data/dfm15/native-reasoning-v2/sampled_dfm15/metadata.json')
        tokenizer = Path(metadata['tokenizer_info']['tokenizer_path']).resolve()
        template = Path(metadata['tokenizer_info']['chat_template_path']).resolve()
        results = []
        for language in ('faroese', 'icelandic'):
            name = language+'-dyna-instruct'
            repo = 'danish-foundation-models/'+name
            folder = root/name
            folder.mkdir(exist_ok=True)
            pin = folder/'revision.json'
            if not pin.exists():
                write_json(pin, dict(repo_id=repo, revision=HfApi().dataset_info(repo).sha))
            revision = load(pin)['revision']
            print('Downloading', repo, revision, flush=True)
            snapshot_download(repo, repo_type='dataset', revision=revision,
                              local_dir=folder/'download', max_workers=8,
                              allow_patterns=['*.md', 'data/**/*.parquet'])
            stage = folder/'input'
            receipt = folder/'conversion.json'
            if not receipt.exists():
                stage.mkdir(exist_ok=True)
                # A failed conversion is rebuilt; tokenization starts only after its receipt.
                for stale in stage.glob('*.jsonl'):
                    stale.unlink()
                rows = excluded = 0
                counts = {}
                handle = None
                try:
                    for source in sorted((folder/'download/data').glob('*/*.parquet')):
                        if source.parent.name == 'fo-instruct':
                            excluded += pq.ParquetFile(source).metadata.num_rows
                            continue
                        counts[source.parent.name] = 0
                        for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
                            for row in batch.to_pylist():
                                messages = row['messages']
                                if (len(messages) != 2 or
                                    [m['role'] for m in messages] != ['user', 'assistant'] or
                                    any(not isinstance(m['content'], str) or not m['content'].strip()
                                        for m in messages)):
                                    raise ValueError(f'Unexpected conversation: {source}: {row.get("id")}')
                                if rows % 10000 == 0:
                                    if handle:
                                        handle.close()
                                    handle = (stage/f'part-{rows//10000:05d}.jsonl').open('w')
                                handle.write(json.dumps(dict(messages=messages, source=row.get('source'),
                                    source_id=row.get('id')), ensure_ascii=False)+'\n')
                                rows += 1
                                counts[source.parent.name] += 1
                finally:
                    if handle:
                        handle.close()
                write_json(receipt, dict(rows=rows, constituents=counts,
                    excluded_fo_instruct_rows=excluded,
                    exclusion_reason='Setur/fo-instruct inherited at repeat 10'))
            print('Tokenizing', name, load(receipt), flush=True)
            subprocess.run([sys.executable, '-m', 'scripts.tokenize_chat_template', str(stage),
                '-o', str(folder/'tokenized'), '--workers', str(args.workers),
                '--tokenizer-path', str(tokenizer), '--chat-template', str(template)], check=True,
                env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1'})
            rows = tokens = over = 0
            for part in (folder/'tokenized').glob('part-*.jsonl'):
                lengths = (np.load(part/'inst_len.npy', mmap_mode='r') +
                           np.load(part/'resp_len.npy', mmap_mode='r'))
                rows += len(lengths)
                tokens += int(lengths.sum())
                over += int((lengths > 4096).sum())
            if rows != load(receipt)['rows']:
                raise ValueError('Tokenization changed row count')
            result = dict(name=name, repo_id=repo, revision=revision, rows=rows, tokens=tokens,
                sequences_over_4096=over, tokenized_path=str(folder/'tokenized'),
                repeat=1, sampled=False, cross_corpus_exact_overlap_checked=False)
            write_json(folder/'prepared.json', result)
            results.append(result)
            write_json(root/'additions.json', dict(additions=results, sampled=False))
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
