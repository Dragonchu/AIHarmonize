"""Base loader"""


class BaseLoader:
    """Base loader"""

    def __init__(self, settings):
        self.settings = settings
        self.setup()

    def setup(self):
        """Setup something when init loader."""

    def load(self, data: str):
        """Write data to loader"""
        raise NotImplementedError()

    def close(self):
        """Close something"""

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self
