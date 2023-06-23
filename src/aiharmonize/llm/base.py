"""Base LLM class."""
from aiharmonize.communication_element.base import CommunicationElement


class BaseLLM:

    def __init__(self, settings) -> None:
        self.settings = settings
        self.setup()

    def setup(self):
        """将LLM塑造为指定的角色"""

    def run(self, element: CommunicationElement):
        """运行LLM"""
        raise NotImplementedError()
