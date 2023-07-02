"""
基于OpenAI实现的AI
"""

import logging
import os
import numpy as np

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

from aiharmonize.harmonizeai.base import BaseHarmonizeAI

logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class Gpt3HarmonizeAI(BaseHarmonizeAI):
    """Gpt3 base AI"""

    def __init__(self, settings):
        super().__init__(settings)
        os.environ["OPENAI_API_KEY"] = self.settings.OPENAI_API_KEY
        self.llm = OpenAI(temperature=0.7)
        self.arch_prompt = PromptTemplate(
            input_variables=["program"],
            # template="From now your are a programmer. How to understand the subclasses in {program}?\n"
            template="From now your are a programmer. What are the innermost subclasses in {program}?\n"
        )
        self.arch_chain = LLMChain(llm=self.llm, prompt=self.arch_prompt)
        self.embedding_model = OpenAIEmbeddings()

    def setup(self):
        """将LLM塑造为指定的角色"""
        logger.info("Setup AI.")

    def transform(self, communication_element):
        """运行LLM"""
        logger.info("AI is running.")
        # return self.arch_chain.run(communication_element)

    def get_subfunc(self, file_path, graph):
        # subfunc_prompt = PromptTemplate(
        #     input_variables=["program"],
        #     template="From now your are a programmer. What are the innermost subclasses in {program}?\n"
        # )
        # subfunc_chain = LLMChain(llm=OpenAI(temperature=1.0), prompt=subfunc_prompt)
        # subfunc_name_str = subfunc_chain.run(graph)
        # subfunc_name_strs = subfunc_name_str.strip().split("\n")[1:]
        subfunc_name_strs = ['CachedCalculator__CachedCalculator____init__', 'CachedCalculator__CachedCalculator__add', 'CachedCalculator__CachedCalculator__divide', 'CachedCalculator__CachedCalculator__multiply', 'CachedCalculator__CachedCalculator__subtract']
        # print(subfunc_name_strs)
        subfunc_strs = [i.split("__")[-1]  for i in subfunc_name_strs if '__init__' not in i]
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
        
        for k, v in subfunc_points.items():
            print("*******************")
            print(k, v)
            subfunc_detail_prompt = PromptTemplate(
                input_variables=["name", "code"],
                template="From now your are a programmer. The code in function {name} is \"{code}\". Please generally describe this function in detail.\n"
            )
            subfunc_detail_chain = LLMChain(llm=OpenAI(temperature=1.0), prompt=subfunc_detail_prompt)
            func_detail = subfunc_detail_chain.run({"name":k, "code":v})
            subfunc_details[k] = func_detail
            print("function name: {0} \n function details: \n {1}".format(k, func_detail))
            
            embedding = self.embedding_model.embed_query(func_detail)
            subfunc_embs[k] = embedding
            # print("function name: {0} \n function embeddings: \n {1}".format(k, embedding))
            print("function name: {0} \n function embeddings: \n {1}".format(k, len(embedding)))
            print("*******************")

        return subfunc_details, subfunc_embs

    def calcu_similarity(self, file_dict):
        """
        Input:  calculate function embedding similarities of C files
        """
        sims_files = {}
        for i, (ki,vi) in enumerate(file_dict.items()):
            for j, (kj,vj) in enumerate(file_dict.items()):
                if i>j:
                    sims, _ = self.calcu_similarity2(vi, vj)
                    sims_files[ki+"_"+kj] = sims

        print(sims_files)

    def calcu_similarity2(self, emb_dict1, emb_dict2):
        """
        Input: calculate N/M function embedding similarities
        """
        sims = np.ones((len(emb_dict1.keys()), len(emb_dict2.keys())))
        sim_func_names = [i+"_"+j for i in list(emb_dict1.keys()) for j in list(emb_dict2.keys())]
        for i, (ki,vi) in enumerate(emb_dict1.items()):
            for j, (kj,vj) in enumerate(emb_dict2.items()):
                if i>=j:
                    sims[i][j] = np.dot(vi, vj) / ( np.linalg.norm(vi) * np.linalg.norm(vj))
        print(sims, sim_func_names)
        return sims, sim_func_names
