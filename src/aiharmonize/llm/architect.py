"""架构师AI"""
from aiharmonize.communication_element.base import CommunicationElement
from aiharmonize.llm.base import BaseLLM


class Architect(BaseLLM):

    def run(self, element: CommunicationElement):
        print("架构师AI运行中...")
