"""CPU-only HumanEval rescoring with unchanged outputs, tests and timeout.

Uses the existing approved local execution backend; this is not an isolation
sandbox. Never use this utility for untrusted arbitrary archives.
Original archives remain untouched. Publication is a separate explicit step.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import anyio
from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.scorer import accuracy, scorer, stderr
from inspect_ai.util._sandbox.context import (
    cleanup_sandbox_environments_sample,
    init_sandbox_environments_sample,
)
from inspect_ai.util._sandbox.local import LocalSandboxEnvironment
from dfm_evals.tasks.code import verify_sanitized as production_scorer


@scorer(metrics=[accuracy(), stderr()])
def verify_sanitized():
    delegate = production_scorer(max_concurrent_verifications=2)
    slots = anyio.Semaphore(2)

    async def run(state, target):
        async with slots:
            envs = await init_sandbox_environments_sample(
                LocalSandboxEnvironment, 'humaneval-reverify', None, {}, None, state.metadata)
            try:
                return await delegate(state, target)
            finally:
                await cleanup_sandbox_environments_sample(
                    'local', 'humaneval-reverify', None, envs, False)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OPENAI_API_KEY', 'unused-no-inference')
    paths = sorted(args.root.glob('shard_*/*/inspect/*.eval'))
    if len(paths) != 4:
        raise ValueError(f'Expected four unambiguous HumanEval archives, found {len(paths)}')
    receipt = []
    ids = set()
    for path in paths:
        log = read_eval_log(path)
        assert log.status == 'success' and log.eval.task == 'dfm_evals/humaneval'
        assert log.eval.sandbox.type == 'local'
        for sample in log.samples:
            assert sample.id not in ids
            ids.add(sample.id)
        original_outputs = {s.id: s.output.model_dump() for s in log.samples}
        rescored = score(log, verify_sanitized(), action='overwrite', display='plain')
        assert {s.id: s.output.model_dump() for s in rescored.samples} == original_outputs
        timeouts = sum('timed out' in s.scores['verify_sanitized'].explanation.lower()
                       for s in rescored.samples)
        dest = args.output/path.name
        write_eval_log(rescored, str(dest))
        item = dict(source=str(path), output=str(dest),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    samples=len(rescored.samples), timeouts=timeouts,
                    correct=sum(s.scores['verify_sanitized'].value == 'C' for s in rescored.samples))
        receipt.append(item)
        print(json.dumps(item), flush=True)
    assert len(ids) == 164
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if any(r['timeouts'] for r in receipt):
        raise RuntimeError('Verification timeouts remain: inspect before publishing')


if __name__ == '__main__':
    main()
