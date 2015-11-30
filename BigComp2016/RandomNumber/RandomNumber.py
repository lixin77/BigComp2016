__author__ = 'lixin77'
# -*-coding:utf-8-*-
import random

def RandInt(beg, end):
    """
    生成[beg, end]区间内的随机整数
    """
    return random.randint(beg, end)


def RandFloat(begin=0, end=1):
    """
    生成(begin, end)范围内的浮点数
    """
    return random.uniform(begin, end)