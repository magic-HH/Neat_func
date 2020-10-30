# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 10:49
# @Author  : HH
# @File    : ActiveFunc.py
# @Project : ConvTest

import math

"""
   激活函数
"""


def tanh(x):
    return math.tanh(x)


def sigmoid(x):
    y = 1 / (1 + pow(math.e, -4.9 * x))
    return y


func_active = {"sigmoid": sigmoid, "tanh": tanh}
