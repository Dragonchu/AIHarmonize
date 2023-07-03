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
        """Run manage"""
        # print(__file__)
        # def find_functions(files):
        #     with ZipFile("tmp.zip", "w") as zip_obj:
        #         #pylint: disable=unused-variable
        #         for idx, file in enumerate(files):
        #             zip_obj.write(file.name, file.name.split("/")[-1])
        #     return "tmp.zip"
        # demo = gr.Interface(
        #     find_functions,
        #     gr.File(file_count="multiple", file_types=["text", ".json", ".py"]),
        #     "file",
        #     examples=[[[os.path.join(os.path.dirname(__file__), "examples/CachedCalculator.py"),
        #     os.path.join(os.path.dirname(__file__), "examples/FileOutputCalculator.py")]]],
        #     cache_examples=True
        # )
        # demo.launch()
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
                        ,inputs=file)
            # 显示功能点的按钮
            find_func_btn = gr.Button("Find Functions")
            # 执行计划的按钮
            gen_plan_btn = gr.Button("Generate Plan")
            # 交互
            find_func_btn.click(self.find_functions,inputs=file,outputs=functions_text)
            gen_plan_btn.click(gen_plan,inputs=functions_text,outputs=plan_text)
        demo.queue(concurrency_count=5, max_size=20).launch()
        # with self.extractor_kls(settings) as extractor:
        #     with self.loader_kls(settings) as loader:
        #         self.harmonize(extractor, loader)
        # logger.info('Exit example_etl.')

    def find_functions(self, tmp_files):
        """获取功能点"""
        path = settings.FILE_EXTRACTOR_PATH
        for file in tmp_files:
            file_path = os.path.join(path,file.name.split("/")[-1])
            print(file_path)
            with open(file_path, "w+",encoding=DEFAULT_ENCODING) as f_o:
                f_o.write(file.read().decode("utf-8"))
        logger.info(settings.TRANSFORMER_NAME, settings.EXTRACTOR_NAME)
        with self.extractor_kls(settings) as extractor:
            with self.loader_kls(settings) as loader:
                self.harmonize(extractor, loader)
        logger.info('Exit example_etl.')
        res_file = open(settings.FILE_LOADER_PATH, 'r', encoding=DEFAULT_ENCODING)
        content = res_file.readlines()
        res_file.close()
        return content
    
    def harmonize(self, extractor: BaseExtractor, loader: BaseLoader):
        """Transform data from extractor to loader."""
        logger.info('Start transformer data ......')

        details, embs = {}, {}
        for i, (file_path, graph) in enumerate(extractor.extract().items()):
            loader.load(file_path+"\n")
            loader.load("graph:\n"+graph+"\nfunctions:\n")
            print(file_path, graph)
            data = self.harmonizeai.get_subfunc(file_path, graph)
            # data = self.harmonizeai.transform(i)
            details[file_path], embs[file_path] = data[0], data[1]
            loader.load_dict(data[0])
        sims_files, sims_names = self.harmonizeai.calcu_similarity(embs)
        merge_funcs = self.harmonizeai.merge_method(sims_files, sims_names)
        loader.load_dict(merge_funcs)
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
