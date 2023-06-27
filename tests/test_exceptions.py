"""Test exception"""
from aiharmonize.exceptions import PluginNotFoundError


def test_plugin_not_found_error():
    """test plugin not found error"""
    error = PluginNotFoundError('foo', 'bar')
    assert str(error) == 'Can not found "bar" plugin in foo'
