"""
基于OpenAI实现的AI
"""

import logging

from aiharmonize.harmonizeai.base import BaseHarmonizeAI

logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class Gpt3HarmonizeAI(BaseHarmonizeAI):
    """Gpt3 base AI"""
    def transform(self, communication_element):
        """运行LLM"""
        logger.info("AI is running.")
        return communication_element
