"""AI之间交流的基本元素"""


class CommunicationElement:
    """AI之间交流的基本元素"""

    def __init__(self, type_: str):
        self.__type = type_
        
    @property
    def type(self) -> str:
        """元素类型"""
        return self.__type
