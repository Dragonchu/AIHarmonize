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
import pyan

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
        logger.warn("Welcome to HarmonizeAI!")
        logger.debug("Start to extract data.")
        print("Welcome to HarmonizeAI!")
        demo = gr.Blocks()
        with demo:
            with gr.Row():
                with gr.Column():
                    # 上传文件
                    file = gr.File(file_count="multiple", file_types=["text", ".json", ".py"])
                    # 测试用例
                    gr.Examples(examples=[[[os.path.join(os.path.dirname(__file__), "examples/CachedCalculator.py"),
                                            os.path.join(os.path.dirname(__file__), "examples/FileOutputCalculator.py")]]]
                                , inputs=file)
            with gr.Row():
                with gr.Column():
                    # 显示功能点的按钮
                    find_func_btn = gr.Button("Find Functions")
                    # 显示功能点并交给用户修改
                    functions_text = gr.Textbox()
                with gr.Column():
                    # 计算相似度的按钮
                    calcu_sim_btn = gr.Button("Calculate Similarity")
                    # 显示相似度
                    sim_text = gr.Textbox()
            with gr.Row():
                with gr.Column():
                    # 执行计划的按钮
                    gen_plan_btn = gr.Button("Generate Plan")
                    # 显示架构师AI的执行计划
                    plan_text = gr.Textbox()
            with gr.Row():
                with gr.Column():
                    # 生成合并后文件的按钮
                    gen_file_btn = gr.Button("Generate File")
                    # 显示合并后的文件
                    merged_file = gr.Textbox()
            with gr.Row():
                with gr.Column():
                    # 生成测试用例的按钮
                    gen_test_btn = gr.Button("Generate Test")
                    # 显示测试用例
                    test_text = gr.Textbox()

            # 交互
            find_func_btn.click(self.gen_fp, inputs=file, outputs=functions_text)
            calcu_sim_btn.click(self.calcu_sim, inputs=[file], outputs=sim_text)
            gen_plan_btn.click(self.gen_plan, inputs=[file, functions_text], outputs=plan_text)
            gen_file_btn.click(self.gen_file, inputs=[file, plan_text], outputs=merged_file)
            gen_test_btn.click(self.gen_test, inputs=[file], outputs=test_text)
        demo.queue().launch(debug=True)

    def gen_fp(self, files):
        """获取功能点"""
        res = ""
        for idx, temp_file in enumerate(files):
            with open(temp_file.name, encoding=DEFAULT_ENCODING) as fo:
                output = self.harmonizeai.transform("fp_bot", fo.read())
                # output = "save money."
                # print(output)
                res += output
                print(res)
        return res

    def gen_plan(self, files, user_decide):
        """架构师AI生成执行计划"""
        return self.harmonizeai.transform("plan_bot", split_json_string(user_decide))

    def gen_file(self, files, plan):
        """架构师AI生成合并后的文件"""
        communication_element = {"plan": plan}
        for idx, temp_file in enumerate(files):
            with open(temp_file.name, encoding=DEFAULT_ENCODING) as fo:
                communication_element["file" + str(idx)] = fo.read()
        return self.harmonizeai.transform("merge_bot", communication_element)

    def gen_test(self, files):
        """生成测试用例"""
        communication_element = {}
        with open("/tmp/merged_file.py", encoding=DEFAULT_ENCODING) as fo:
            communication_element["test_file"] = fo.read()
        for idx, temp_file in enumerate(files):
            with open(temp_file.name, encoding=DEFAULT_ENCODING) as fo:
                communication_element["file" + str(idx)] = fo.read()
        return self.harmonizeai.transform("test_bot", communication_element)

    def calcu_sim(self, files):
        """计算相似度"""
        func_graphs = {}
        details, embs = {}, {}
        res = ""
        for file in files:
            with open(file.name, encoding=DEFAULT_ENCODING) as fo:
                graph = pyan.create_callgraph(filenames=file.name, format="dot", grouped_alt=True)
                func_graphs[file.name] = graph
                data = self.harmonizeai.get_subfunc(fo)
                details[file.name], embs[file.name] = data[0], data[1]
                res += self.harmonizeai.calcu_similarity(embs)
        return res

def split_json_string(json_string):
    # 查找两个JSON文件的分隔位置
    separator = '}{'
    index = json_string.find(separator)

    # 拆分成两个JSON字符串
    json1 = json_string[:index + 1]
    json2 = json_string[index + 1:]

    # 解析JSON字符串为JSON对象
    json_list = []
    json_list.append(json1)
    json_list.append(json2)

    return json_list


def get_extension(namespace: str, name: str):
    """Get extension by name from namespace."""
    extension_manager = ExtensionManager(namespace=namespace, invoke_on_load=False)
    for ext in extension_manager.extensions:
        if ext.name == name:
            logger.info('Load plugin: %s in namespace "%s"', ext.plugin, namespace)
            return ext.plugin
    raise PluginNotFoundError(namespace=namespace, name=name)
