"""Manage"""
import logging
from typing import Type

from stevedore import ExtensionManager

from aiharmonize.config import settings
from aiharmonize.exceptions import PluginNotFoundError
from aiharmonize.extractor.base import BaseExtractor
from aiharmonize.harmonizeai.base import BaseHarmonizeAI
from aiharmonize.loader.base import BaseLoader
import gradio as gr

logger = logging.getLogger(__name__)


class Manage:
    """Manager"""

    def __init__(self):
        self.extractor_kls: Type[BaseExtractor] = get_extension(
            'aiharmonize.extractor',
            settings.EXTRACTOR_NAME,
        )
        self.loader_kls: Type[BaseLoader] = get_extension(
            'aiharmonize.loader',
            settings.LOADER_NAME,
        )
        self.harmonizeai_kls: Type[BaseHarmonizeAI] = get_extension(
            'aiharmonize.harmonizeai',
            settings.TRANSFORMER_NAME,
        )

        self.harmonizeai: BaseHarmonizeAI = self.harmonizeai_kls(settings)

    def run(self):
        """Run manage"""
        demo = gr.Interface(fn=greet, inputs="text", outputs="text")
        demo.launch()
        with self.extractor_kls(settings) as extractor:
            with self.loader_kls(settings) as loader:
                self.harmonize(extractor, loader)
        logger.info('Exit example_etl.')

    def harmonize(self, extractor: BaseExtractor, loader: BaseLoader):
        """Transform data from extractor to loader."""
        logger.info('Start transformer data ......')
        for i in extractor.extract():
            data = self.harmonizeai.transform(i)
            loader.load(data)

        logger.info('Data processed.')


def get_extension(namespace: str, name: str):
    """Get extension by name from namespace."""
    extension_manager = ExtensionManager(namespace=namespace, invoke_on_load=False)
    for ext in extension_manager.extensions:
        if ext.name == name:
            logger.info('Load plugin: %s in namespace "%s"', ext.plugin, namespace)
            return ext.plugin

    raise PluginNotFoundError(namespace=namespace, name=name)

def greet(name):
    return "Hello " + name + "!"
