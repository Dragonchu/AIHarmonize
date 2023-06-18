"""Base LLM class."""


class BaseLLM:
    """"Base LLM class."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.setup()

    def setup(self):
        """将LLM塑造为指定的角色"""

    def run(self, *args, **kwargs):
        """运行LLM"""
        raise NotImplementedError()
