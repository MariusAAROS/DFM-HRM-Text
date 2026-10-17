import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from condition_dfm15_math_answers import condition


def test_preserves_reasoning_modes():
    prompt = '<bos><|turn>user\nCompute 78 + 39.<turn|>\n<|turn>model\n'
    for prefix in ('78 + 39 = 117. ', '<|channel>thought\n78 + 39 = 117.<channel|>'):
        result, _ = condition(prompt, prefix + r'\boxed{117}' + '<turn|>\n')
        assert result is not None
        p, r, answer = result
        assert 'ANSWER: <number>' in p
        assert r.startswith(prefix.rstrip())
        assert r.endswith('ANSWER: 117<turn|>\n')
        assert answer == '117'


def test_numeric_finals_with_punctuation_and_direct_answers():
    prompt = '<|turn>user\nThere are only three apples. How many apples?<turn|>\n<|turn>model\n'
    for target in (r'Final answer: \boxed{3}.', r'Thus $\boxed{3}$.', '3',
                   '<|channel>thought\nCount three.<channel|>3', 'Work.\n#### 3'):
        result, _ = condition(prompt, target + '<turn|>\n')
        assert result is not None
        assert result[2] == '3'
        assert result[1].endswith('ANSWER: 3<turn|>\n')
        if target.startswith('Final') or target.startswith('Thus'):
            assert result[1].startswith(target)


def test_no_guessing_from_numbers_or_changing_symbolic_answers():
    prompt = '<|turn>user\nCompute the result.<turn|>\n<|turn>model\n'
    for target in (r'\boxed{3} or 4.', r'\boxed{3} percent.',
                   r'\boxed{\frac{1}{2}}.', 'There are 3 apples.',
                   r'\boxed{3} + 4', '<|channel>thought\n3'):
        assert condition(prompt, target + '<turn|>\n')[0] is None


def test_preserves_constraints_and_ambiguous_answers():
    for question in ('Use boxed format.', 'Return only the answer.', 'Prove this theorem.'):
        prompt = '<|turn>user\n' + question + '<turn|>\n<|turn>model\n'
        assert condition(prompt, r'\boxed{117}<turn|>')[0] is None
    prompt = '<|turn>user\nCompute the result.<turn|>\n<|turn>model\n'
    for response in (r'\boxed{1/2}<turn|>', 'It is probably 117.<turn|>',
                     r'<|channel>thought\boxed{117}<turn|>'):
        assert condition(prompt, response)[0] is None


def test_publish_preserves_unchanged_rows_and_recomputes_totals(tmp_path, monkeypatch):
    import json
    import numpy as np
    import condition_dfm15_math_answers as c

    target, work = tmp_path/'sample', tmp_path/'work'
    target.mkdir(); work.mkdir()
    monkeypatch.setattr(c, 'TARGET', target)
    monkeypatch.setattr(c, 'WORK', work)
    np.save(target/'tokens.npy', np.arange(100, dtype=np.int32))
    np.save(work/'tokens.npy', np.arange(8, dtype=np.int32))
    old = np.array([[1, 2, 3, 2]], dtype=np.uint64)
    new = np.array([[100, 4, 104, 4]], dtype=np.uint64)
    untouched = np.array([[10, 2, 12, 2]], dtype=np.uint64)
    np.savez(work/'mapping.npz', old=old, new=new)
    (work/'before.json').write_text(json.dumps({'stored_tokens': 100}))
    (work/'preparation.json').write_text(json.dumps({'version': 2, 'changed_unique': 1}))
    (target/'metadata.json').write_text('{}')
    for epoch in range(3):
        folder = target/f'epoch_{epoch}'; folder.mkdir()
        for i, field in enumerate(c.FIELDS):
            np.save(folder/(field+'.npy'), np.concatenate([old, untouched])[:, i])
    c.publish()
    for epoch in range(3):
        got = np.column_stack([np.load(target/f'epoch_{epoch}'/(k+'.npy')) for k in c.FIELDS])
        np.testing.assert_array_equal(got, np.concatenate([new, untouched]))
        assert not list((target/f'epoch_{epoch}').glob('*.conditioning.npy'))
    assert json.loads((target/'metadata.json').read_text())['total_length'] == 12
    assert json.loads((target/'math-answer-conditioning-verified.json').read_text())['version'] == 2
