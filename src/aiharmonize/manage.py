"""Manage"""
import logging
import os
from typing import Type

import gradio as gr
from stevedore import ExtensionManager

from aiharmonize.config import settings
from aiharmonize.exceptions import PluginNotFoundError
from aiharmonize.extractor.base import BaseExtractor
from aiharmonize.harmonizeai.base import BaseHarmonizeAI
from aiharmonize.loader.base import BaseLoader

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
        demo = gr.Blocks()
        with demo:
            # 上传文件
            file = gr.File(file_count="single", file_types=["text", ".json", ".py"])
            # 显示功能点并交给用户修改
            functions_text = gr.Textbox()
            # 显示架构师AI的执行计划
            plan_text = gr.Textbox()
            
            # 测试用例
            gr.Examples(examples=[[[os.path.join(os.path.dirname(__file__), "examples/CachedCalculator.py")]]],inputs=file)
            
            # 显示功能点的按钮
            find_func_btn = gr.Button("Find Functions")
            # 执行计划的按钮
            gen_plan_btn = gr.Button("Generate Plan")
            
            # 交互
            find_func_btn.click(find_functions,inputs=file,outputs=functions_text)
            gen_plan_btn.click(gen_plan,inputs=functions_text,outputs=plan_text)
        demo.queue(concurrency_count=5, max_size=20).launch()
        # with self.extractor_kls(settings) as extractor:
        #     with self.loader_kls(settings) as loader:
        #         self.harmonize(extractor, loader)
        # logger.info('Exit example_etl.')

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

def gen_plan(user_decide):
    """架构师AI生成执行计划"""
    return f"Here we go: \n {user_decide}"

def find_functions(tmp_file):
    """获取功能点"""
    res = ""
    with open(tmp_file.name, encoding='utf8') as file:
        for line in file.readlines():
            res += line
    return res
