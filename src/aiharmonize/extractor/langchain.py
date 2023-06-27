"""
langchain extractor

extract data using lanchain.
"""
import logging

from aiharmonize.constants import DEFAULT_ENCODING
from aiharmonize.extractor.base import BaseExtractor

logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class LangchainExtractor(BaseExtractor):
    """Langchain extractor"""

    def extract(self):
        """使用Langchain工具读取待融合文件"""
        logger.info("Extracting data using langchain.")
        extractor_path = self.settings.FILE_EXTRACTOR_PATH
        logger.info('Extract data from %s', extractor_path)
        with open(extractor_path, 'r', encoding=DEFAULT_ENCODING) as file:
            for i in file:
                yield i
