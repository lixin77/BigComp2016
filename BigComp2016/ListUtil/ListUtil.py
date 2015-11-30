__author__ = 'lixin77'
# -*-coding:utf-8-*-

import string
import math

def Normalize(list, smoother=0.0):
    """
    对向量list进行归一化处理，得到每个元素出现的概率
    :param list: 向量
    :param smoother: 平滑值，缺省值为0; 为了防止0概率的出现
    """
    sum = Sum(list)
    K = len(list)
    newlist = []
    if sum > 0:
        newlist = [float((item + smoother) / (sum + K * smoother)) for item in list]
    return newlist


def Sum(list):
    """
    计算list中所有元素的和
    """
    res = 0
    for item in list:
        res += item
    return res


def Initial(size, data=object):
    """
    生成一个大小为size, 所有元素都为data的列表
    :param size: 列表大小
    :param data: 列表元素
    """
    list = []
    for i in xrange(size):
        list.append(data)
    return list


def InitialMat(M, N, data=0):
    """
    初始化大小为M * N的矩阵，所有元素初始化为data
    :param M:
    :param N:
    :param data: 矩阵元素
    """
    mat = []
    for i in xrange(M):
        row = Initial(N, data)
        mat.append(row)
    return mat

def InitialEmptyMat(rows):
    """
    初始化一个空的matrix
    :param rows:
    """
    mat = []
    for i in xrange(rows):
        tmp = []   #代表每一个文档包含的词，初始化为空
        mat.append(tmp)
    return mat

def toString(list):
    """
    将list中的元素拼接成字符串
    方便用作文件操作
    :param list: 列表元素
    """
    listStr = ""
    count = 0
    for ele in list:
        if type(ele) == int:
            eleStr = str(ele)
        elif type(ele) == float:
            #浮点数转换为字符串，保留8位小数
            eleStr = str("%.10f"%ele)
        elif type(ele) == str or type(ele) == unicode:
            eleStr = ele
        if count != len(list) - 1:
            eleStr += " "
        count += 1
        listStr += eleStr
    listStr += "\n"
    return listStr

def StringToFloatList(SS):
    """
    string 转换为float
    :param SS: 从文件中读取的字符串
    """
    res = [string.atof(item) for item in SS.split(" ")]
    return res


def AssignList(LL):
    """
    将LL中的值拷贝到另一个list中
    :param LL: 字符串
    """
    newLL = []
    for ele in LL:
        newLL.append(ele)
    return newLL


def FindMax(LL):
    """
    返回列表LL中最大的元素
    """
    LL.sort()
    return LL[len(LL) - 1]


def ListCopy(LL):
    """
    return the copy of the list LL
    """
    assert isinstance(LL, list)
    newLL = []
    for ele in LL:
        newLL.append(ele)
    return newLL


def CaculateCosine(lista, listb):
    """
    compute the cosine similarities between the lista and listb
    """
    assert isinstance(lista, list) and isinstance(listb, list) and len(lista) == len(listb)
    anorm = NORM(lista)
    bnorm = NORM(listb)
    length = len(lista)
    total = 0.0
    for i in xrange(length):
        total += (lista[i] * listb[i])
    return total / (anorm * bnorm)


def NORM(ll):
    """
    compute the length of the list/vector ll
    """
    assert isinstance(ll, list)
    total = 0.0
    for ele in ll:
        total += (ele * ele)
    dist = math.sqrt(total)
    return dist

def findMaxPos(ll):
    """
    return the index of the maximum element
    """
    assert isinstance(ll, list)
    MAX = -1
    MAX_POS = 0
    length = len(ll)
    for i in xrange(length):
        if ll[i] > MAX:
            MAX_POS = i
            MAX = ll[i]
    return MAX_POS

def findTrueLabel(ss):
    """
    find the true label according to the ratings
    """
    assert isinstance(ss, str)
    contents = ss.split(' ')
    Max = -1
    res = []
    for content in contents:
        label, count = content.split(':')
        count_int = int(count)
        #if label == 'Total':
        if label == 'all':
            continue
        if count_int > Max:
            Max = count_int
            res[:] = []
            res.append(label)
        elif count_int == Max:
            res.append(label)
    return res