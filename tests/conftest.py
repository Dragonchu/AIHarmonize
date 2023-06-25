"""Test config"""
import tempfile

import pytest
from click.testing import CliRunner


@pytest.fixture()
def clicker():
    """clicker fixture"""
    yield CliRunner()


@pytest.fixture()
def foo_file():
    """foo file"""
    with tempfile.NamedTemporaryFile(mode='w') as file:
        file.write('foo')
        file.flush()
        yield file.name
