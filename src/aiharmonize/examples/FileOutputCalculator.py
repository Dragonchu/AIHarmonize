# 实现一个可以将结果输出到文件的计算器类，可以对两个数进行加、减、乘、除四则运算，并将结果输出到指定的文件中

class FileOutputCalculator:
    def __init__(self, output_file):
        self.output_file = output_file

    def add(self, a, b):
        result = a + b
        with open(self.output_file, "a") as f:
            f.write(f"{a} + {b} = {result}\n")
        return result

    def subtract(self, a, b):
        result = a - b
        with open(self.output_file, "a") as f:
            f.write(f"{a} - {b} = {result}\n")
        return result

    def multiply(self, a, b):
        result = a * b
        with open(self.output_file, "a") as f:
            f.write(f"{a} * {b} = {result}\n")
        return result

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        with open(self.output_file, "a") as f:
            f.write(f"{a} / {b} = {result}\n")
        return result

# 使用示例
calc = FileOutputCalculator("output.txt")
print(calc.add(2, 3))  # 输出 5，并将结果输出到 output.txt 文件中
print(calc.multiply(2, 3))  # 输出 6，并将结果输出到 output.txt 文件中