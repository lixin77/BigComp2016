__author__ = 'lixin77'
# -*-coding:utf-8-*-

from RandomNumber import RandomNumber


def UniSample(K):
    """
    产生从O到K－1的整数
    :param K: 主题个数
    """
    return RandomNumber.RandInt(0, K - 1)


def MultSample(ProbList):
    """
    从多项分布ProbList中采样, ProbList表示剔除当前btm之后的主题分布
    :param ProbList: 多项分布
    """
    size = len(ProbList)
    for i in xrange(1, size):
        ProbList[i] += ProbList[i - 1]
    #随机产生一个［0，1）的小数
    u = RandomNumber.RandFloat()
    res = 0
    for k in xrange(size):
        if ProbList[k] >= u * ProbList[size - 1]:
            #抽样结果
            res = k
            break
    #res为抽样后的主题编号
    return res

