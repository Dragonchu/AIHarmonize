"""Base harmonizeai class."""


class BaseHarmonizeAI:
    """Base HarmonizeAI class."""
    def __init__(self, settings) -> None:
        self.settings = settings
        self.setup()

    def setup(self):
        """将LLM塑造为指定的角色"""

    def transform(self, role, communication_element):
        """运行LLM"""
        raise NotImplementedError()

    def get_subfunc(self, fo):
        pass

    def calcu_similarity(self, embs):
        pass
