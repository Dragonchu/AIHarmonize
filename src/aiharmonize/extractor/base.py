"""Base extractor."""


class BaseExtractor:
    """Base extractor"""

    def __init__(self, settings):
        self.settings = settings
        self.setup()

    def setup(self):
        """Setup something when init extractor"""

    def extract(self):
        """Extract data."""
        raise NotImplementedError()

    def close(self):
        """Close something."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
