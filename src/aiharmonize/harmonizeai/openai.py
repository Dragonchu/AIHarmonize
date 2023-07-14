"""
基于OpenAI实现的AI
"""

import logging
import os

import numpy as np
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from aiharmonize.harmonizeai.base import BaseHarmonizeAI
from aiharmonize.harmonizeai.communication_element import FunctionPoint, FunctionPoints

logger = logging.getLogger(__name__)

get_function_point_template = """You are a program that provides function point descriptions based on the input code file.
The term "input" refers to any information you receive, and if the input is not code, your response must be "Input is an unrecognized code file." 
The term "output" refers to the function points in the code file, where only methods that can be accessed by external programs are considered as function points.
Private methods and constructors are not considered as function points. 
{format_instructions}
"""

gen_project_merge_plan_template = """
You are a program responsible for merging function points. Your input will be fuction points from two files. 
The fuction points definitions for each file are JSON files, in which the "weight" attribute indicates the importance of the entire functionality point. 
The range for the "weight" value is from 0 to 1, with higher values indicating that the functionality point should be retained as much as possible, and lower values suggesting that the fuction points can be considered for merging with other points or even discarded. 
A "weight" value of 1 means that the functionality point must exist independently, while a "weight" value of 0 means that the functionality point should be discarded.
Your output should directly provide the merged result of the two functionalities, in a format consistent with the input fuction points JSON. 
You have the flexibility to decide the class name of the output based on the input.
{format_instructions}
"""

# pylint: disable=too-few-public-methods


class Gpt3HarmonizeAI(BaseHarmonizeAI):
    """Gpt3 base AI"""

    def __init__(self, settings):
        super().__init__(settings)
        os.environ["OPENAI_API_KEY"] = self.settings.OPENAI_API_KEY
        self.fp_bot_prompt = None
        self.fp_bot = None
        self.setup_fp_bot()
        self.setup_plan_bot()

    def setup(self):
        """将LLM塑造为指定的角色"""
        logger.info("Setup AI.")

    def setup_fp_bot(self):
        """设置获取功能点的AI"""
        fp_parser = PydanticOutputParser(pydantic_object=FunctionPoints)
        fp_system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=get_function_point_template,
                input_variables=[],
                partial_variables={"format_instructions": fp_parser.get_format_instructions()},
            )
        )
        file_input_template = "{file}"
        file_input_prompt = HumanMessagePromptTemplate.from_template(file_input_template)
        self.fp_bot_prompt = ChatPromptTemplate.from_messages([fp_system_message_prompt, file_input_prompt])
        
    def setup_plan_bot(self):
        gen_plan_parser = PydanticOutputParser(pydantic_object=FunctionPoint)
        gen_plan_system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=gen_project_merge_plan_template,
                input_variables=[],
                partial_variables={"format_instructions": gen_plan_parser.get_format_instructions()},
            )
        )
        gen_plan_input_template = "Here are the function points of the two files:{file}\nPlease merge them into one file."
        gen_plan_input_prompt = HumanMessagePromptTemplate.from_template(gen_plan_input_template)
        self.gen_plan_bot_prompt = ChatPromptTemplate.from_messages([gen_plan_system_message_prompt, gen_plan_input_prompt])

    def transform(self, role, communication_element):
        """运行LLM"""
        if role == "fp_bot":
            _input = self.fp_bot_prompt.format_prompt(file=communication_element)
            fp_bot = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, verbose=True)
            output = fp_bot(_input.to_string())
            return output
        elif role == "plan_bot":
            _input = self.gen_plan_bot_prompt.format_prompt(file=communication_element)
            plan_bot = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, verbose=True)
            output = plan_bot(_input.to_string())
            return output
