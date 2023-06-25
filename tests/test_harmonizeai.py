"""Test transformer"""
import pytest

from aiharmonize.harmonizeai.base import BaseHarmonizeAI
from aiharmonize.harmonizeai.openai import Gpt3HarmonizeAI


def test_base_process(mocker):
    """Test base transformer"""
    process = BaseHarmonizeAI(mocker.MagicMock())
    with pytest.raises(NotImplementedError):
        process.transform('foo')


@pytest.mark.parametrize(
    'data, expect_value',
    [
        ('xx', 'xx'),
    ]
)
def test_strip_process(mocker, data, expect_value):
    """Test strip transformer"""
    processor = Gpt3HarmonizeAI(mocker.MagicMock())
    res = processor.transform(data)
    assert res == expect_value
