"""执行者AI与测试者AI的交流元素"""

from aiharmonize.communication_element.base import CommunicationElement


class Executor2Testor(CommunicationElement):
    """执行者AI与测试者AI的交流元素"""

    def __init__(self, type_: str):
        super().__init__(type_)
