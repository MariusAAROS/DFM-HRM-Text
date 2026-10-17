"""Prepare new training-only GSM-Symbolic and Danish correction examples."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from dfm12.io import load, lock, write_json


def convert(name, row):
    if name == 'gsm-symbolic-en-train':
        reasoning, sep, answer = row['answer'].rpartition('####')
        if not sep or answer.strip() != str(row['target']).strip():
            raise ValueError('Unverified GSM final answer')
        reasoning = re.sub(r'<<[^<>]*>>', '', reasoning).strip()
        messages = [dict(role='user', content=row['question']),
                    dict(role='assistant', reasoning_content=reasoning,
                         content='\\boxed{'+answer.strip()+'}')]
    else:
        messages = [dict(role='user', content='Ret den sproglige fejl i denne sætning. '
                         'Svar kun med den rettede sætning:\n\n'+row['bad sentence']),
                    dict(role='assistant', content=row['good sentence'])]
    if any(not m['content'].strip() for m in messages):
        raise ValueError('Empty message')
    return dict(messages=messages)


def main():
    root = Path('data/dfm15/small-additions').resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = [('gsm-symbolic-en-train', 'multilingual-gsm-symbolic',
             'data/eng/train_synthetic-00000-of-00001.parquet', True),
            ('linguistic-quality-da', 'linguistic-quality', 'train.parquet', False)]
    meta = load('data/dfm15/native-reasoning-v2/sampled_dfm15/metadata.json')['tokenizer_info']
    with lock(root/'.lock'):
        results = []
        for name, repo_name, filename, thinking in spec:
            repo = 'danish-foundation-models/'+repo_name
            folder = root/name
            folder.mkdir(exist_ok=True)
            pin = folder/'revision.json'
            if not pin.exists():
                write_json(pin, dict(repo_id=repo, revision=HfApi().dataset_info(repo).sha))
            rev = load(pin)['revision']
            source = hf_hub_download(repo, filename, repo_type='dataset', revision=rev,
                                     local_dir=folder/'download')
            stage = folder/'input'
            stage.mkdir(exist_ok=True)
            seen = set()
            duplicates = 0
            with (stage/'train.jsonl').open('w') as out:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        value = convert(name, row)
                        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
                        digest = hashlib.sha256(key.encode()).hexdigest()
                        if digest in seen:
                            duplicates += 1
                            continue
                        seen.add(digest)
                        out.write(key+'\n')
            cmd = [sys.executable, '-m', 'scripts.tokenize_chat_template', str(stage),
                '-o', str(folder/'tokenized'), '--tokenizer-path',
                str(Path(meta['tokenizer_path']).resolve()), '--chat-template',
                meta['chat_template_path'], '--workers', '1']
            if thinking:
                cmd.append('--enable-thinking')
            subprocess.run(cmd, check=True, env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false'})
            part = folder/'tokenized/train.jsonl'
            lengths = np.load(part/'inst_len.npy')+np.load(part/'resp_len.npy')
            if len(lengths) != len(seen):
                raise ValueError('Tokenization row mismatch')
            result = dict(name=name, repo_id=repo, revision=rev, selected_files=[filename],
                rows=len(lengths), tokens=int(lengths.sum()), internal_duplicates_removed=duplicates,
                sequences_over_4096=int((lengths > 4096).sum()),
                tokenized_path=str(folder/'tokenized'), repeat=1, sampled=False,
                enable_thinking=thinking, cross_corpus_exact_overlap_checked=False)
            results.append(result)
            write_json(folder/'prepared.json', result)
            write_json(root/'additions.json', dict(additions=results, sampled=False))
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
