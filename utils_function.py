import numpy as np
from fractions import Fraction
import math


def round_to_n_significant_digits(num, n):
    """
    保留 n 位有效数字的函数。

    参数:
    - num: 要处理的数字 (float 或 int)。
    - n: 要保留的有效数字位数 (正整数)。

    返回:
    - 保留 n 位有效数字的数字。
    """
    if num == 0:
        return 0  # 对于 0，直接返回 0
    else:
        # 计算有效数字所需的小数位数
        shift = n - int(math.floor(math.log10(abs(num)))) - 1  # 数字的数量级调整
        return round(num, shift)


def approximate_with_rational(matrix, precision):
    """
    使用连分数逼近量化矩阵中的浮点数，动态调整分母限制。

    参数:
        matrix (np.ndarray): 原始浮点数矩阵，可以是多维矩阵。
        max_denominator (int): 逼近的分母最大值。默认值为100。

    返回:
        np.ndarray: 量化后的矩阵，元素为 Fraction 类型。
    """

    # 映射函数：根据数值的大小映射出对应的最大分母限制
    def map_to_max_denominator(value):
        if value == 0: return 1
        abs_value = abs(value)
        d = 10 ** precision / abs_value
        d = int(d)
        return d

    def quantize_element(element):
        if isinstance(element, np.ndarray):  # 如果是子数组，递归处理
            return np.array([quantize_element(sub_elem) for sub_elem in element], dtype=object)
        else:
            # 使用映射函数来动态获取该元素的最大分母
            dynamic_denominator = map_to_max_denominator(element)
            # 量化该元素
            return Fraction(str(element)).limit_denominator(dynamic_denominator)

    return quantize_element(matrix)


# 递归地将量化后的权重矩阵中的每个 Fraction 转换为浮点数
def convert_fractions_to_floats(element, precision):
    if isinstance(element, np.ndarray):  # 如果是子数组，递归处理
        return np.array([convert_fractions_to_floats(sub_elem, precision) for sub_elem in element])
    elif isinstance(element, Fraction):  # 如果是分数，转换为浮点数
        return round_to_n_significant_digits(float(element), precision)
    else:
        raise TypeError


# 并行处理函数
def process_layer(key, value, precision):
    try:
        print(f"层名称: {key}")
        quantized_weights = approximate_with_rational(value, precision)
        float_weights = convert_fractions_to_floats(quantized_weights)
        return key, quantized_weights, float_weights
    except Exception as e:
        print(f"处理层 {key} 时发生错误: {e}")
        raise
