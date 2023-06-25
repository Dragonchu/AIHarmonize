"""Test loader"""
import tempfile
from pathlib import Path

import pytest

from aiharmonize.loader.base import BaseLoader
from aiharmonize.loader.file import FileLoader


def test_base_dest(mocker):
    """Test base loader"""
    close_mock = mocker.patch.object(BaseLoader, 'close')
    with BaseLoader(mocker.MagicMock()) as base:
        with pytest.raises(NotImplementedError):
            base.load('foo')
    assert close_mock.called_once()


def test_file_dest(mocker):
    """Test file loader"""
    with tempfile.NamedTemporaryFile() as file:
        settings_mock = mocker.MagicMock()
        settings_mock.FILE_LOADER_PATH = file.name
        with FileLoader(settings_mock) as loader:
            loader.load('foo')
        file = Path(file.name)
        stat = file.stat()
        assert stat.st_size == 3
