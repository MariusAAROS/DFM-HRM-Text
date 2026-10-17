import pytest

from scripts.prepare_dfm15_small_additions import convert
from scripts.register_dfm15_additions import fingerprint


def test_gsm_separates_reasoning_and_final():
    row = dict(question='What is 2+2?', answer='2+2 = <<2+2=4>>4.\n#### 4', target='4')
    assistant = convert('gsm-symbolic-en-train', row)['messages'][-1]
    assert assistant == dict(role='assistant', reasoning_content='2+2 = 4.', content='\\boxed{4}')


def test_gsm_rejects_mismatching_target():
    with pytest.raises(ValueError):
        convert('gsm-symbolic-en-train', dict(question='Q', answer='R\n#### 4', target='5'))


def test_correction_preserves_target():
    row = {'bad sentence': 'Bad', 'good sentence': 'Good'}
    assert convert('linguistic-quality-da', row)['messages'][-1]['content'] == 'Good'


def test_fingerprint_ignores_metadata_not_answers():
    a = [dict(role='user', content=' Q ', id='one'), dict(role='assistant', content='A')]
    b = [dict(role='user', content='Q'), dict(role='assistant', content='A')]
    assert fingerprint(a) == fingerprint(b)
    b[-1]['content'] = 'B'
    assert fingerprint(a) != fingerprint(b)
