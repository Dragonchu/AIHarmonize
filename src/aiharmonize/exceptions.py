"""Exception"""


class AiHarmonizeError(Exception):
    """AiHarmonize error"""


class PluginNotFoundError(AiHarmonizeError):
    """PluginNotFoundError"""

    def __init__(self, namespace: str, name: str):
        super().__init__()
        self._namespace = namespace
        self._name = name

    def __repr__(self):
        return f'Can not found "{self._name}" plugin in {self._namespace}'

    def __str__(self):
        return self.__repr__()
