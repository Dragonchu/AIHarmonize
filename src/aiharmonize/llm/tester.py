"""测试者AI"""
from aiharmonize.communication_element.base import CommunicationElement
from aiharmonize.llm.base import BaseLLM


class Tester(BaseLLM):
    """测试者AI"""

    def run(self, element: CommunicationElement):
        print("测试者AI运行中...")
