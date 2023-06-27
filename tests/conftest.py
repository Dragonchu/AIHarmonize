"""Test config"""
import tempfile

import pytest
from click.testing import CliRunner


@pytest.fixture()
def clicker():
    """clicker fixture"""
    yield CliRunner()


@pytest.fixture()
def foo_file():
    """foo file"""
    with tempfile.NamedTemporaryFile(mode='w') as file:
        file.write('foo')
        file.flush()
        yield file.name


@pytest.fixture()
def cached_calculator():
    """cached calculator"""
    with tempfile.NamedTemporaryFile(mode='w') as file:
        file.write("# 实现一个带缓存的计算器类，可以对两个数进行加、减、乘、除四则运算，并缓存上一次计算结果\n"
                   "\n"
                   "class CachedCalculator:\n"
                   "    def __init__(self):\n"
                   "        self.cache = {}\n"
                   "\n"
                   "    def add(self, a, b):\n"
                   "        key = f\"add-{a}-{b}\"\n"
                   "        if key in self.cache:\n"
                   "            return self.cache[key]\n"
                   "        result = a + b\n"
                   "        self.cache[key] = result\n"
                   "        return result\n"
                   "\n"
                   "    def subtract(self, a, b):\n"
                   "        key = f\"subtract-{a}-{b}\"\n"
                   "        if key in self.cache:\n"
                   "            return self.cache[key]\n"
                   "        result = a - b\n"
                   "        self.cache[key] = result\n"
                   "        return result\n"
                   "\n"
                   "    def multiply(self, a, b):\n"
                   "        key = f\"multiply-{a}-{b}\"\n"
                   "        if key in self.cache:\n"
                   "            return self.cache[key]\n"
                   "        result = a * b\n"
                   "        self.cache[key] = result\n"
                   "        return result\n"
                   "\n"
                   "    def divide(self, a, b):\n"
                   "        if b == 0:\n"
                   "            raise ValueError(\"Cannot divide by zero\")\n"
                   "        key = f\"divide-{a}-{b}\"\n"
                   "        if key in self.cache:\n"
                   "            return self.cache[key]\n"
                   "        result = a / b\n"
                   "        self.cache[key] = result\n"
                   "        return result\n"
                   "\n"
                   "# 使用示例\n"
                   "calc = CachedCalculator()\n"
                   "print(calc.add(2, 3))  # 输出 5，结果被缓存\n"
                   "print(calc.add(2, 3))  # 输出 5，直接从缓存中取得结果\n"
                   "print(calc.multiply(2, 3))  # 输出 6，结果被缓存\n")
        file.flush()
        yield file.name


@pytest.fixture()
def file_output_calculator():
    """file output calculator"""
    with tempfile.NamedTemporaryFile(mode='w') as file:
        file.write("# 实现一个可以将结果输出到文件的计算器类，可以对两个数进行加、减、乘、除四则运算，并将结果输出到指定的文件中\n"
                   "\n"
                   "class FileOutputCalculator:\n"
                   "    def __init__(self, output_file):\n"
                   "        self.output_file = output_file\n"
                   "\n"
                   "    def add(self, a, b):\n"
                   "        result = a + b\n"
                   "        with open(self.output_file, \"a\") as f:\n"
                   "            f.write(f\"{a} + {b} = {result}\n\")\n"
                   "        return result\n"
                   "\n"
                   "    def subtract(self, a, b):\n"
                   "        result = a - b\n"
                   "        with open(self.output_file, \"a\") as f:\n"
                   "            f.write(f\"{a} - {b} = {result}\n\")\n"
                   "        return result\n"
                   "\n"
                   "    def multiply(self, a, b):\n"
                   "        result = a * b\n"
                   "        with open(self.output_file, \"a\") as f:\n"
                   "            f.write(f\"{a} * {b} = {result}\n\")\n"
                   "        return result\n"
                   "\n"
                   "    def divide(self, a, b):\n"
                   "        if b == 0:\n"
                   "            raise ValueError(\"Cannot divide by zero\")\n"
                   "        result = a / b\n"
                   "        with open(self.output_file, \"a\") as f:\n"
                   "            f.write(f\"{a} / {b} = {result}\n\")\n"
                   "        return result\n"
                   "\n"
                   "# 使用示例\n"
                   "calc = FileOutputCalculator(\"output.txt\")\n"
                   "print(calc.add(2, 3))  # 输出 5，并将结果输出到 output.txt 文件中\n"
                   "print(calc.multiply(2, 3))  # 输出 6，并将结果输出到 output.txt 文件中\n")
        file.flush()
        yield file.name
