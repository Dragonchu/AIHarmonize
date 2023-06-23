"""测试者AI与架构师AI的交流元素"""

from aiharmonize.communication_element.base import CommunicationElement


class Tester2Architect(CommunicationElement):
    """测试者AI与架构师AI的交流元素"""

    def __init__(self, type_: str):
        super().__init__(type_)
