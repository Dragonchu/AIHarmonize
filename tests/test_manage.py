"""Test manage"""
import pytest

from aiharmonize.exceptions import PluginNotFoundError
from aiharmonize.extractor.langchain import LangchainExtractor
from aiharmonize.manage import Manage, get_extension


def test_get_extension():
    """Test get extension"""
    plugin = get_extension('aiharmonize.extractor', 'langchain')
    assert plugin is LangchainExtractor


def test_get_extension_error():
    """Test get extension error"""
    with pytest.raises(PluginNotFoundError):
        get_extension('aiharmonize.extractor', 'xxx')


def test_manage_run(mocker):
    """Test manage run"""
    mocker.patch('aiharmonize.manage.get_extension')
    process_mock = mocker.patch.object(Manage, 'harmonize')
    manage = Manage()

    manage.run()
    assert process_mock.called_once()


def test_manage_transform(mocker):
    """Test manage transform"""
    magic_mock = mocker.MagicMock()
    manage = Manage()
    # 将manage的harmonizeai属性赋值为mock对象
    manage.harmonizeai = magic_mock
    magic_mock.extract.return_value = [1, 2]
    manage.harmonize(magic_mock, magic_mock)

    # 测试mock对象的方法是否被调用
    assert magic_mock.extract.called_once()
    assert magic_mock.load.call_count == 2
    assert magic_mock.transform.call_count == 2
