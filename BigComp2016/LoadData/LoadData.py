__author__ = 'lixin77'
# -*-coding:utf-8-*-

import os
import string


def LoadDataFromFile(path):
    """
    :param path:短文本存放路径
    """
    print "Begin to load data from file..."
    #转换为绝对路径
    fp = open(path, 'r')
    Docs = []
    EmotionRating = []
    Ratings = string
    for line in fp:
        #去掉结尾换行符
        ll = line.strip('\n').strip('\r')
        items = ll.split("\t")
        print items
        #情感投票字符串
        EmotionRating.append(items[1])
        Docs.append(items[2])
    fp.close()
    print "Done, load ", len(Docs), " docs from the file"
    #print "Done!"
    return Docs, EmotionRating


def LoadStopWords():
    """
    从指定路径读取停用词表
    return:停用词列表
    """
    path = os.getcwd()
    path += "/StopWords.txt"
    fp = open(path, 'r')
    #获取停用词列表
    StopWordsList = [line.strip('\n') for line in fp]
    fp.close()
    return StopWordsList


def LoadDictionary():
    """
    从指定路径加载训练词典
    """
    path = os.getcwd() + "/dictionary.txt"
    fp = open(path, 'r')
    Dictionary = dict()
    for line in fp:
        elements = line.strip('\n').split(" ")
        #词的id
        k = string.atoi(elements[0])
        #词本身
        v = elements[1]
        Dictionary[k] = v
    fp.close()
    return Dictionary


def LoadEmotionRating():
    """
    从指定文件读取情感投票
    """
    print "Begin to load the emotion ratings..."
    path = os.getcwd() + "/ratings.txt"
    fp = open(path, 'r')
    EmotionRating = []
    for line in fp:
        line = line.strip('\n')
        EmotionRating.append(line)
    fp.close()
    print "Done!!"
    return EmotionRating
