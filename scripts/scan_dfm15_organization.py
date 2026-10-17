"""Pin the DFM organization inventory and record constituent-level overlap evidence."""
import json
from pathlib import Path
from huggingface_hub import HfApi
from dfm12.io import write_json


def decision(name):
    if name in {'faroese-dyna-instruct', 'icelandic-dyna-instruct', 'linguistic-quality'}:
        return 'prepared_addition', 'Pinned CPU-converted/tokenized addition; see config/dfm15_sources.json'
    if name == 'multilingual-gsm-symbolic':
        return 'train_synthetic_only', 'English synthetic train only; original train duplicates GSM8k; all test splits excluded'
    if name in {'dialog-k2-da', 'instruction-dataset-v1'}:
        return 'blocked_private_storage', 'Hub returned 403 organization private storage limit; schema visible but content inaccessible'
    if name in {'dfm-dyna-instruct', 'norwegian-dyna-instruct'}:
        return 'unchanged_composite', 'Current revision and constituent hashes equal October 5 scan; retain previous subset policy'
    if name in {'ifeval-da', 'multi-ifeval', 'global-piqa-da'} or name.startswith('dala'):
        return 'preserve_train_eval_boundary', 'Do not add benchmark copies; retain inherited held-out and training policies'
    if name == 'Icelandic-flan':
        return 'no_data_files', 'Repository has no Parquet/JSONL payload; Icelandic FLAN content supplied through dyna-instruct'
    if name.startswith('croco-'):
        return 'retain_existing_selection', 'Existing simpo-full-50k route and earlier near-total prompt overlap decision retained'
    if name == 'danish-wildchat4.8M':
        return 'retain_exclusion', 'Previously explicitly excluded; no new blanket admission'
    if name in {'danish-gigaword', 'nasjonalt-vitenarkiv', 'cellar-metadata', 'dynaword-annotations'} or name.endswith('-dynaword'):
        return 'grounding_not_direct_sft', 'Raw corpus/metadata; not silently added as instruction supervision; transformations are separate work'
    if name in {'ai_arena_udtraek', 'ai-arenaen', 'ai-arenaen-raw', 'laerebogen', 'kaenguruen', 'synthetic-values-model-charter'}:
        return 'retain_inherited_routes', 'Existing converted/audited routes; no blanket duplicate admission; not a row-level freshness claim'
    return 'needs_review', 'No automatic admission'


def main():
    api = HfApi()
    old = {x['repo']: x for x in json.loads(Path(
        'docs/reports/dyna-source-scan-20261005/inventory.json').read_text())}
    rows = []
    for dataset in api.list_datasets(author='danish-foundation-models'):
        info = api.dataset_info(dataset.id, files_metadata=True)
        files = []
        baseline = old.get(dataset.id, {})
        for f in info.siblings:
            if not f.rfilename.endswith(('.parquet', '.jsonl', '.jsonl.gz')):
                continue
            sha = getattr(f.lfs, 'sha256', None)
            prior = baseline.get('current_files', {}).get(f.rfilename, {})
            files.append(dict(path=f.rfilename, bytes=f.size, sha256=sha,
                unchanged_from_oct5=bool(sha and sha == prior.get('sha256'))))
        status, reason = decision(dataset.id.split('/')[-1])
        rows.append(dict(repo_id=dataset.id, revision=info.sha, private=info.private,
            decision=status, reason=reason,
            baseline_revision=baseline.get('revision'), files=files,
            card=info.card_data.to_dict() if info.card_data else {}))
        write_json('data/dfm15/organization-inventory.json', rows)
        print(dataset.id, len(files), 'files', flush=True)


if __name__ == '__main__':
    main()
