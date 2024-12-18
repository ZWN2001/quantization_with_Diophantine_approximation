import torch
import torch.nn as nn
from fractions import Fraction


# 有理数类：自定义有理数计算
class RationalNumber:
    def __init__(self, numerator, denominator):
        self.value = Fraction(numerator, denominator)

    def __repr__(self):
        return str(self.value)

    # 加法
    def __add__(self, other):
        return RationalNumber(
            self.value.numerator * other.value.denominator + other.value.numerator * self.value.denominator,
            self.value.denominator * other.value.denominator)

    # 乘法
    def __mul__(self, other):
        return RationalNumber(self.value.numerator * other.value.numerator,
                              self.value.denominator * other.value.denominator)

    # 转为浮点数
    def to_float(self):
        return float(self.value)


# 有理数矩阵
def generate_rational_matrix(shape=(4, 4)):
    # 使用有理数初始化矩阵
    return [[RationalNumber(int(100 * i * j), 100) for j in range(1, shape[1] + 1)] for i in range(1, shape[0] + 1)]


# 简单的神经网络
class RationalNN(nn.Module):
    def __init__(self):
        super(RationalNN, self).__init__()
        self.fc1 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc1(x)
        return x


# 将有理数应用到网络的权重
def apply_rational_weights_to_nn(model, rational_weights):
    with torch.no_grad():
        # 转换有理数矩阵为浮点数矩阵，直接替换原始权重
        float_weights = torch.tensor([[r.to_float() for r in row] for row in rational_weights], dtype=torch.float32)
        model.fc1.weight.copy_(float_weights)
    return model


# 测试
rational_weights = generate_rational_matrix()
print("Rational Matrix Weights (Rational Numbers):")
for row in rational_weights:
    print(row)

# 打印有理数矩阵
print("Rational Matrix Weights (Rational Numbers):")
for row in rational_weights:
    print(row)

# 创建一个简单神经网络
model = RationalNN()

# 将有理数权重应用到神经网络中
model = apply_rational_weights_to_nn(model, rational_weights)

# 打印更新后的权重
print("\nUpdated model weights (after applying rational approximation):")
print(model.fc1.weight)
