"""
基于OpenAI实现的AI
"""

import logging
import os

from langchain import LLMChain, OpenAI, PromptTemplate

from aiharmonize.harmonizeai.base import BaseHarmonizeAI

logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class Gpt3HarmonizeAI(BaseHarmonizeAI):
    """Gpt3 base AI"""

    def __init__(self, settings):
        super().__init__(settings)
        os.environ["OPENAI_API_KEY"] = self.settings.OPENAI_API_KEY
        self.llm = OpenAI(temperature=0.7)
        self.prompt = PromptTemplate(
            input_variables=["program"],
            template="From now your are a programmer.Do you like {program}?\n"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def setup(self):
        """将LLM塑造为指定的角色"""
        logger.info("Setup AI.")

    def transform(self, communication_element):
        """运行LLM"""
        logger.info("AI is running.")
        return self.chain.run("apple")
