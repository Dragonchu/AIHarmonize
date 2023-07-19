"""
基于OpenAI实现的AI
"""

import logging
import os
import subprocess

import numpy as np
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from aiharmonize.harmonizeai.base import BaseHarmonizeAI
from aiharmonize.harmonizeai.communication_element import FunctionPoints, MergePlan, TestPlan

logger = logging.getLogger(__name__)

get_function_point_template = """You are a program that provides function point descriptions based on the input code file.
The term "input" refers to any information you receive, and if the input is not code, your response must be "Input is an unrecognized code file." 
The term "output" refers to the function points in the code file, where only methods that can be accessed by external programs are considered as function points.
Private methods and constructors are not considered as function points. 
{format_instructions}
"""

gen_project_merge_plan_template = """
As a program responsible for merging function points, your input will consist of function points from two files, provided as JSON files.
Each function point definition will have a "weight" attribute representing its importance.
The "retain" value determines whether the function point is retained after merging.
Your task is to merge the function points.
The merged result should conform to the format of the input function points JSON files. 
Keep in mind that the merging process should focus on the function level, allowing for decomposition of functions and the creation of new functions with equivalent functionality.
The desired output is to have fewer and more concise merged functions.
Simply combining the functions from the two input files into one file is not considered a good output.
The function name and file name you output can be decided by yourself, and should not be limited to the input file.
You are free to abstract common methods that exist in multiple functions. Remember, the JSON file you receive is an abstraction of an actual program.
For example, if there is a calculator class with cached methods for addition, subtraction, multiplication, and division, and another calculator class that outputs the values to a file for each method, it can be considered to merge these two calculator classes into one.
The merged calculator class would provide the four arithmetic methods (addition, subtraction, multiplication, and division) and also provide an additional parameter to control whether the results are written to the cache or to a file.
{format_instructions}
"""

merge_class_template = """
You are a program that merges two python classes into one, your output should only contain the merged python class, 
no other output is required. Your merge needs to comply with certain requirements, this is your merge requirement: {requirements}, 
these are the two classes you need to merge.
All files will be placed in the same folder, and each class will be in a separate file.
The file name will be the class name followed by the .py extension.
For example, the CachedCalculator class should be in the CachedCalculator.py file.
Please make sure to correctly reference the module in the output file.
The first class: {class1}\n
The second class: {class2}\n
"""

test_plan_template = """
You are a test developer.
Next, you will be provided with all the files in a folder.
Please use Python to write a test class that tests the specified Python file, and generate a shell command that will use the test file you created.
You should write a test class that tests the third file, and generate a shell command that will use the test file you created.
All files will be placed in the same folder, and each class will be in a separate file.
The file name will be the class name followed by the .py extension.
For example, the CachedCalculator class should be in the CachedCalculator.py file.
Please make sure to correctly reference the module in the output file.
{format_instructions}
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
        self.setup_merge_bot()
        self.setup_test_bot()

        self.llm = OpenAI(temperature=0.0)
        self.arch_prompt = PromptTemplate(
            input_variables=["program"],
            template="From now your are a programmer. What are the innermost subclasses in {program}?\n"
        )
        self.arch_chain = LLMChain(llm=self.llm, prompt=self.arch_prompt)
        self.embedding_model = OpenAIEmbeddings()

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

    def setup_test_bot(self):
        test_plan_parser = PydanticOutputParser(pydantic_object=TestPlan)
        test_plan_system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=test_plan_template,
                input_variables=[],
                partial_variables={"format_instructions": test_plan_parser.get_format_instructions()},
            )
        )
        test_plan_input_template = """This is first file: {file1}. \n This is second file: {file2}. \n This is third file: {file3}.
        """
        test_plan_input_prompt = HumanMessagePromptTemplate.from_template(test_plan_input_template)
        self.test_plan_bot_prompt = ChatPromptTemplate.from_messages([test_plan_system_message_prompt, test_plan_input_prompt])

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
            merged_file_path = os.path.join("/tmp/MergedCalculator.py")
            with open(merged_file_path, "w", encoding="utf-8") as f:
                f.write(output)
            return output
        elif role == "test_bot":
            _input = self.test_plan_bot_prompt.format_prompt(file1=communication_element["file0"], file2=communication_element["file1"], file3=communication_element["test_file"])
            test_bot = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, verbose=True)
            output = test_bot(_input.to_string())
            return output
    
    def get_subfunc(self, file):
        subfunc_name_strs = ['CachedCalculator__CachedCalculator____init__', 'CachedCalculator__CachedCalculator__add',
                             'CachedCalculator__CachedCalculator__divide',
                             'CachedCalculator__CachedCalculator__multiply',
                             'CachedCalculator__CachedCalculator__subtract']
        subfunc_strs = [i.split("__")[-1] for i in subfunc_name_strs if '__init__' not in i]
        substart = [0] * len(subfunc_strs)
        subfunc_points, subfunc_details, subfunc_embs = {}, {}, {}
        for i in subfunc_strs:
            subfunc_points[i] = ""
            subfunc_details[i] = ""
            subfunc_embs[i] = ""

        ## TODO: How to set an adaptive chunck_size??
        # with open(file_path, 'r') as f:
        #     codes = f.read()
        #     python_splitter = RecursiveCharacterTextSplitter.from_language(
        #         language=Language.PYTHON, chunk_size=200, chunk_overlap=0
        #     )
        #     python_docs = python_splitter.create_documents([codes])
        #     for i in python_docs:
        #         print(i)

        line = file.readline()
        while line is not None and line != '':
            for i in range(len(subfunc_strs)):
                if " " + subfunc_strs[i] + "(" in line and substart[i] == 0:
                    substart[i] += 1
                    subfunc_points[subfunc_strs[i]] += line
                elif substart[i] > 0 and "def" not in line:
                    substart[i] += 1
                    subfunc_points[subfunc_strs[i]] += line
                elif substart[i] == 0:
                    continue
                else:
                    substart[i] = 0
                    continue
            line = file.readline()

        for k, v in subfunc_points.items():
            logger.debug("*******************")
            logger.debug("k: {0}, v: {1}".format(k, v))
            subfunc_detail_prompt = PromptTemplate(
                input_variables=["name", "code"],
                template="From now your are a programmer. The code in function {name} is \"{code}\". Please generally describe this function in detail.\n"
            )
            subfunc_detail_chain = LLMChain(llm=OpenAI(temperature=0.0), prompt=subfunc_detail_prompt)
            func_detail = subfunc_detail_chain.run({"name": k, "code": v})
            subfunc_details[k] = func_detail
            logger.debug("function name: {0} \n function details: \n {1}".format(k, func_detail))

            embedding = self.embedding_model.embed_query(func_detail)
            subfunc_embs[k] = embedding
            logger.debug("function name: {0} \n function embeddings: \n {1}".format(k, len(embedding)))
            logger.debug("*******************")

        return subfunc_details, subfunc_embs

    def calcu_similarity(self, file_dict):
        """
        Input:  calculate function embedding similarities of C files
        """
        sims_files = {}
        sims_func_names = {}
        for i, (ki, vi) in enumerate(file_dict.items()):
            for j, (kj, vj) in enumerate(file_dict.items()):
                if i > j:
                    sims, sim_func_names = self.calcu_similarity2(vi, vj)
                    sims_files[ki + "_" + kj] = sims
                    sims_func_names[ki + "_" + kj] = sim_func_names
        return str(sims_files)+ "\n" + str(sims_func_names)

    def calcu_similarity2(self, emb_dict1, emb_dict2):
        """
        Input: calculate N/M function embedding similarities
        """
        sims = np.ones((len(emb_dict1.keys()), len(emb_dict2.keys())))
        sim_func_names = [i + "_" + j for i in list(emb_dict1.keys()) for j in list(emb_dict2.keys())]
        for i, (ki, vi) in enumerate(emb_dict1.items()):
            for j, (kj, vj) in enumerate(emb_dict2.items()):
                if i >= j:
                    sims[i][j] = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
        return sims, sim_func_names
