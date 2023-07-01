# 实现一个带缓存的计算器类，可以对两个数进行加、减、乘、除四则运算，并缓存上一次计算结果

class CachedCalculator:
    def __init__(self):
        self.cache = {}

    def add(self, a, b):
        key = f"add-{a}-{b}"
        if key in self.cache:
            return self.cache[key]
        result = a + b
        self.cache[key] = result
        return result

    def subtract(self, a, b):
        key = f"subtract-{a}-{b}"
        if key in self.cache:
            return self.cache[key]
        result = a - b
        self.cache[key] = result
        return result

    def multiply(self, a, b):
        key = f"multiply-{a}-{b}"
        if key in self.cache:
            return self.cache[key]
        result = a * b
        self.cache[key] = result
        return result

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        key = f"divide-{a}-{b}"
        if key in self.cache:
            return self.cache[key]
        result = a / b
        self.cache[key] = result
        return result

# 使用示例
calc = CachedCalculator()
print(calc.add(2, 3))  # 输出 5，结果被缓存
print(calc.add(2, 3))  # 输出 5，直接从缓存中取得结果
print(calc.multiply(2, 3))  # 输出 6，结果被缓存
