"""Manage"""
import logging
import os
from typing import Type

import gradio as gr
from stevedore import ExtensionManager

from aiharmonize.config import settings
from aiharmonize.constants import DEFAULT_ENCODING
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
        demo = gr.Blocks()
        with demo:
            # 上传文件
            file = gr.File(file_count="multiple", file_types=["text", ".json", ".py"])
            # 显示功能点并交给用户修改
            functions_text = gr.Textbox()
            # 显示架构师AI的执行计划
            plan_text = gr.Textbox()
            # 测试用例
            gr.Examples(examples=[[[os.path.join(os.path.dirname(__file__), "examples/CachedCalculator.py"),
                                    os.path.join(os.path.dirname(__file__), "examples/FileOutputCalculator.py")]]]
                        , inputs=file)
            # 显示功能点的按钮
            find_func_btn = gr.Button("Find Functions")
            # 执行计划的按钮
            gen_plan_btn = gr.Button("Generate Plan")
            # 交互
            find_func_btn.click(self.gen_fp, inputs=file, outputs=functions_text)
            gen_plan_btn.click(self.gen_plan, inputs=[file ,functions_text], outputs=plan_text)
        demo.queue().launch(debug=True)

    def gen_fp(self, files):
        """获取功能点"""
        res = ""
        for idx, temp_file in enumerate(files):
            with open(temp_file.name, encoding=DEFAULT_ENCODING) as fo:
                output = self.harmonizeai.transform("fp_bot", fo.read())
                #output = "save money."
                # print(output)
                res += output
                print(res)
        return res

    def gen_plan(self, files, user_decide):
        """架构师AI生成执行计划"""
        return self.harmonizeai.transform("plan_bot", user_decide)

def get_extension(namespace: str, name: str):
    """Get extension by name from namespace."""
    extension_manager = ExtensionManager(namespace=namespace, invoke_on_load=False)
    for ext in extension_manager.extensions:
        if ext.name == name:
            logger.info('Load plugin: %s in namespace "%s"', ext.plugin, namespace)
            return ext.plugin
    raise PluginNotFoundError(namespace=namespace, name=name)
