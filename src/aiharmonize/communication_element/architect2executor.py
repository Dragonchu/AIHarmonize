"""架构师AI与执行者AI的交流元素"""

from aiharmonize.communication_element.base import CommunicationElement


class Architect2Executor(CommunicationElement):
    """架构师AI与执行者AI的交流元素"""

    def __init__(self, type_: str):
        super().__init__(type_)
