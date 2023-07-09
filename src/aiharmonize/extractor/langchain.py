"""
langchain extractor

extract data using lanchain.
"""
import logging
import os

# import pyan

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

        # zip_path = os.path.abspath("./{0}.zip".format(self.settings.FILE_EXTRACTOR_PATH))
        path = self.settings.FILE_EXTRACTOR_PATH
        # cmd = "unzip {0} -d {1}".format(zip_path, path)
        # print(cmd)
        # os.system(cmd)
        # files= os.listdir(path)
        # print(files)
        files = ['CachedCalculator.py', 'FileOutputCalculator.py']

        func_graphs = {}
        for file_name in files:
            file_path = os.path.join(path, file_name)
            file_dot_path = os.path.join(path, file_name.replace(".py", ".dot"))
            print(file_path)
            # graph = pyan.create_callgraph(filenames=file_path, format="dot", grouped_alt=True)
            # with open(file_dot_path, "w+") as f_o:
            #     f_o.write(a)
            # func_graphs[file_path] = graph

        return func_graphs
            
