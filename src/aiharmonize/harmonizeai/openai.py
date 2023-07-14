"""
基于OpenAI实现的AI
"""

import logging
import os

from langchain import OpenAI, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from aiharmonize.harmonizeai.base import BaseHarmonizeAI
from aiharmonize.harmonizeai.communication_element import FunctionPoints, MergePlan

logger = logging.getLogger(__name__)

get_function_point_template = """You are a program that provides function point descriptions based on the input code file.
The term "input" refers to any information you receive, and if the input is not code, your response must be "Input is an unrecognized code file." 
The term "output" refers to the function points in the code file, where only methods that can be accessed by external programs are considered as function points.
Private methods and constructors are not considered as function points. 
{format_instructions}
"""

gen_project_merge_plan_template = """
You are now an architect, and you only focus on the overall framework and core functionality of the program, without paying attention to the specific code details.
You will be given two JSON files that represent two Python classes. 
Your goal is to merge these two Python classes into one class. 
However, as an architect, you don't need to make specific code changes. 
You only need to give abstract instructions on which class's functionality should be kept, and which methods should be abstracted.
Your task is to merge the classes in the most elegant way possible. 
Since these classes have duplicate functionalities, the duplicated functionality should be abstracted as a common feature. 
As for the slightly different parts between the two classes, you can freely decide whether to keep them or simply delete them.
{format_instructions}
"""

merge_class_template = """
You are a program that merges two python classes into one, your output should only contain the merged python class, 
no other output is required. Your merge needs to comply with certain requirements, this is your merge requirement: {requirements}, 
these are the two classes you need to merge.
The first class: {class1}\n
The second class: {class2}\n
"""

# pylint: disable=too-few-public-methods


class Gpt3HarmonizeAI(BaseHarmonizeAI):
    """Gpt3 base AI"""

    def __init__(self, settings):
        super().__init__(settings)
        self.merge_bot_prompt = None
        self.gen_plan_bot_prompt = None
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
        gen_plan_parser = PydanticOutputParser(pydantic_object=MergePlan)
        gen_plan_system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=gen_project_merge_plan_template,
                input_variables=[],
                partial_variables={"format_instructions": gen_plan_parser.get_format_instructions()},
            )
        )
        gen_plan_input_template = "This is first json: {file1}. \n This is second json: {file2}."
        gen_plan_input_prompt = HumanMessagePromptTemplate.from_template(gen_plan_input_template)
        self.gen_plan_bot_prompt = ChatPromptTemplate.from_messages([gen_plan_system_message_prompt, gen_plan_input_prompt])

    def setup_merge_bot(self):
        self.merge_bot_prompt = PromptTemplate.from_template(merge_class_template)

    def transform(self, role, communication_element):
        """运行LLM"""
        if role == "fp_bot":
            _input = self.fp_bot_prompt.format_prompt(file=communication_element)
            fp_bot = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, verbose=True)
            output = fp_bot(_input.to_string())
            return output
        elif role == "plan_bot":
            _input = self.gen_plan_bot_prompt.format_prompt(file1=communication_element[0], file2=communication_element[1])
            plan_bot = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, verbose=True)
            output = plan_bot(_input.to_string())
            return output
        elif role == "merge_bot":
            _input = self.merge_bot_prompt.format_prompt(requirements=communication_element["plan"],
                                                         class1=communication_element["file0"],
                                                         class2=communication_element["file1"])
            merge_bot = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, verbose=True)
            output = merge_bot(_input.to_string())
            return output
