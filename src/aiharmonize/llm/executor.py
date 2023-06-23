"""工作者AI"""
from aiharmonize.communication_element.base import CommunicationElement
from aiharmonize.llm.base import BaseLLM


class Executor(BaseLLM):
    """工作者AI"""

    def run(self, element: CommunicationElement) -> CommunicationElement:
        pass
