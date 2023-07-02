"""Test extractor"""
import pytest

from aiharmonize.extractor.base import BaseExtractor
from aiharmonize.extractor.langchain import LangchainExtractor


def test_base_source(mocker):
    """Test base extractor"""
    close_mock = mocker.patch.object(BaseExtractor, 'close')
    with pytest.raises(NotImplementedError):
        with BaseExtractor(mocker.MagicMock()) as base:
            base.extract()
    assert close_mock.called_once()


def test_langchain_source(mocker, foo_file):
    """Test file extractor"""
    extractor = LangchainExtractor(mocker.MagicMock())
    extractor.settings.FILE_EXTRACTOR_PATH = foo_file
    data = list(extractor.extract())
    # assert data == ['foo']
