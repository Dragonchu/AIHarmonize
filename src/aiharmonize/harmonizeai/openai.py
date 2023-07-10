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
from aiharmonize.harmonizeai.communication_element import FunctionPoints

logger = logging.getLogger(__name__)

get_function_point_template = """You are a program that provides function point descriptions based on the input code file.
The term "input" refers to any information you receive, and if the input is not code, your response must be "Input is an unrecognized code file." 
The term "output" refers to the function points in the code file, where only methods that can be accessed by external programs are considered as function points. Private methods and constructors are not considered as function points. 
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
        self.llm = OpenAI(temperature=0.7)
        self.get_function_point_prompt_template = PromptTemplate.from_template(get_function_point_template)
        self.arch_chain = LLMChain(llm=self.llm, prompt=self.get_function_point_prompt_template)
        self.embedding_model = OpenAIEmbeddings()
        # self.memory = ConversationKGMemory(llm=OpenAI(temperature=1.0))
        # ,"file","code"])#, input_key="human_input")
        self.memory = ConversationBufferMemory(
            memory_key="memory", input_key="name")

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
        self.fp_bot = OpenAI(model_name="gpt-3.5-turbo", max_tokens=-1, temperature=0.0, verbose=True)

    def transform(self, role, communication_element):
        """运行LLM"""
        if role == "fp_bot":
            _input = self.fp_bot_prompt.format_prompt(file=communication_element)
            print("_input:", _input.to_string())
            output = self.fp_bot(_input.to_string())
            print("output:", output)
            return output

    def get_subfunc(self, file_path, graph):
        # subfunc_prompt = PromptTemplate(
        #     input_variables=["program"],
        #     template="From now your are a programmer. What are the innermost subclasses in {program}?\n"
        # )
        # subfunc_chain = LLMChain(llm=OpenAI(temperature=1.0), prompt=subfunc_prompt)
        # subfunc_name_str = subfunc_chain.run(graph)
        # subfunc_name_strs = subfunc_name_str.strip().split("\n")[1:]
        subfunc_name_strs = ['CachedCalculator__CachedCalculator____init__', 'CachedCalculator__CachedCalculator__add',
                             'CachedCalculator__CachedCalculator__divide', 'CachedCalculator__CachedCalculator__multiply', 'CachedCalculator__CachedCalculator__subtract']
        # print(subfunc_name_strs)
        subfunc_strs = [i.split("__")[-1]
                        for i in subfunc_name_strs if '__init__' not in i]
        substart = [0] * len(subfunc_strs)
        subfunc_points, subfunc_details, subfunc_embs = {}, {}, {}
        for i in subfunc_strs:
            subfunc_points[i] = ""
            subfunc_details[i] = ""
            subfunc_embs[i] = ""

        # TODO: How to set an adaptive chunck_size??
        # with open(file_path, 'r') as f:
        #     codes = f.read()
        #     python_splitter = RecursiveCharacterTextSplitter.from_language(
        #         language=Language.PYTHON, chunk_size=200, chunk_overlap=0
        #     )
        #     python_docs = python_splitter.create_documents([codes])
        #     for i in python_docs:
        #         print(i)

        with open(file_path, 'r') as f:
            line = f.readline()
            while line is not None and line != '':
                for i in range(len(subfunc_strs)):
                    if " "+subfunc_strs[i]+"(" in line and substart[i] == 0:
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
                line = f.readline()

        subfunc_detail_prompt = PromptTemplate(
            input_variables=["name", "code", "memory"],
            template=get_function_point_template + """
                    {memory}
                    This file {name} is your input: \"{code}\""""
        )
        # subfunc_detail_chain = LLMChain(llm=OpenAI(temperature=1.0), prompt=subfunc_detail_prompt)
        # func_detail = subfunc_detail_chain.run({"name":k, "code":v})
        subfunc_detail_chain = LLMChain(
            llm=OpenAI(temperature=0.0),
            verbose=True,
            prompt=subfunc_detail_prompt,
            memory=self.memory)

        for k, v in subfunc_points.items():
            print("*******************")
            print(k, v)
            func_detail = subfunc_detail_chain(
                {"name": k+" in file "+file_path, "code": v}, return_only_outputs=True)['text']
            subfunc_details[k] = func_detail.replace("AI:", "")
            print("function name: {0} \n function details: \n {1}".format(
                k, func_detail))

            embedding = self.embedding_model.embed_query(func_detail)
            subfunc_embs[k] = embedding
            # print("function name: {0} \n function embeddings: \n {1}".format(k, embedding))
            print("function name: {0} \n function embeddings: \n {1}".format(
                k, len(embedding)))
            print("*******************")

        return subfunc_details, subfunc_embs

    def merge_method(self, sims, sims_names):
        """
        Input:
            sims: dict{file1_file2} , similarity
            sims_names: dict{file1_file2}, "func1_func2"
        """

        subfunc_merge_prompt = PromptTemplate(
            input_variables=["memory", "name", "name1"],
            template="""From now your are a programmer. 
                    {memory}
                    How to merge the function {name} and the function {name1} to one function\n"""
        )
        subfunc_merge_chain = LLMChain(
            llm=OpenAI(temperature=1.0),
            verbose=True,
            prompt=subfunc_merge_prompt,
            memory=self.memory)
        print(self.memory.buffer)

        merge_funcs = {}
        for i, (file_names, func_names) in enumerate(sims_names.items()):
            func_sims = sims[file_names]
            # print(file_names,func_names,func_sims)
            for j in range(len(func_sims)):
                idx = np.argmax(func_sims[j])
                max_sim = func_sims[j][idx]
                if max_sim > 0.85:
                    name1, name2 = func_names[j][idx].split(
                        "_")[0], func_names[j][idx].split("_")[1]
                    file1, file2 = file_names.split(
                        "_")[0], file_names.split("_")[1]
                    print("current two funcs: ", file1 +
                          " "+name1, file2+" "+name2)
                    func_detail = subfunc_merge_chain(
                        {"name": name1+" in file "+file1, "name1": name2+" in file "+file2}, return_only_outputs=True)["text"]
                    merge_funcs["merge two functions: " + file1+" "+name1 +
                                " and " + file2+" "+name2] = func_detail.replace("AI:", "")

        return merge_funcs

    def calcu_similarity(self, file_dict):
        """
        Input:  calculate function embedding similarities of C files
        """
        sims_files = {}
        sims_names = {}
        for i, (ki, vi) in enumerate(file_dict.items()):
            for j, (kj, vj) in enumerate(file_dict.items()):
                if i > j:
                    sims, names = self.calcu_similarity2(vi, vj)
                    sims_files[ki+"_"+kj] = sims
                    sims_names[ki+"_"+kj] = names
        print(sims_files)
        print(sims_names)
        return sims_files, sims_names

    def calcu_similarity2(self, emb_dict1, emb_dict2):
        """
        Input: calculate N/M function embedding similarities
        """
        sims = np.zeros((len(emb_dict1.keys()), len(emb_dict2.keys())))
        # [[i+"_"+j for i in list(emb_dict1.keys())] for j in list(emb_dict2.keys())]
        sim_func_names = []
        for i, (ki, vi) in enumerate(emb_dict1.items()):
            cur_name = []
            for j, (kj, vj) in enumerate(emb_dict2.items()):
                cur_name.append(ki+"_"+kj)
                if i >= j:
                    sims[i][j] = np.dot(
                        vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
            sim_func_names.append(cur_name)
        # print(sims, sim_func_names)
        return sims, sim_func_names
