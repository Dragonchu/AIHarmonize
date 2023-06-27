"""Test transformer"""
import pytest

from aiharmonize.harmonizeai.base import BaseHarmonizeAI
from aiharmonize.harmonizeai.openai import Gpt3HarmonizeAI


def test_base_process(mocker):
    """Test base transformer"""
    process = BaseHarmonizeAI(mocker.MagicMock())
    with pytest.raises(NotImplementedError):
        process.transform('foo')


def test_llm(cached_calculator, mocker):
    """测试大模型集成"""
    settings = mocker.MagicMock()
    settings.OPENAI_API_KEY = "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"
    harmonize_ai = Gpt3HarmonizeAI(settings)
    print(harmonize_ai.transform(cached_calculator))
