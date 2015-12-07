__author__ = 'lixin77'
# -*-coding:utf-8-*-

import string
#from gensim import corpora
from SaveUtil.SaveUtil import *
import os
from nltk.util import ngrams


def PreprocessText(text, StopWordList):
    """
    预处理一篇文本：剔除标点符号，词干化，去停用词
    :param text: 传入的文本，类型为字符串
    :param StopWordList: 停用词表
    """
    WordList = DelPunctuation(text)
    #StemmeredWordList = Stemmer(WordList)
    FilteredWordList = FilterStopWords(StemmeredWordList, StopWordList)
    return FilteredWordList


def DelPunctuation(text):
    """
    剔除文本中的标点符号
    :param text:需要剔除标点符号的文本，类型为字符串
    return:返回文本中的词的序列
    """
    delset = string.punctuation
    #将标点符号转换为空格
    newText = text.translate(None, delset)
    #文本中的词的列表
    WordList = [word for word in newText.split(" ") if word != '' and word != ' ']
    return WordList


def FilterStopWords(WordList, StopWordList):
    """
    返回去停用词后的词表
    :param WordList:
    :param StopWordList:
    """
    FilteredWordList = filter(lambda x: x.lower() not in StopWordList, WordList)
    return FilteredWordList


def Stemmer(WordList):
    """
    对文档的词表进行词干化
    :param WordList:
    """
    """
    stemmer = nltk.LancasterStemmer()
    StemmeredWordList = [stemmer.stem(w) for w in WordList]
    return StemmeredWordList
    """
    pass

def ConstructDictionary(WordListSet):
    """
    根据输入文档集texts构造词典
    :param WordListSet: 文档集对应的词表，WordListSet[i]表示第i篇文档中的词
    """
    print 'Begin to construct the dictionary'
    #previous use
    #res = corpora.Dictionary(WordListSet)
    res = dict()
    invres = dict()
    count = 0
    for wdl in WordListSet:
        for w in wdl:
            if w not in res.iterkeys():
                res[w] = count
                invres[count] = w
                count += 1
    #SaveDictionary(res)
    print "Done!"
    return res, invres


def Word2Id(WordList, Dictionary):
    """
    将词表转换为词典dictionary中的ID
    :param WordList:
    """
    IDList = []
    """
    items = Dictionary.items()
    for word in WordList:
        #遍历字典查找目标项
        for (k, v) in items:
            if k == word:
                IDList.append(v)
    #print "length of the id list is:", len(IDList)
    """
    for word in WordList:
        IDList.append(Dictionary[word])
    return IDList


def ExtractNGram(sentence, N=2):
    """

    :param sentence: sentence to be processed, stop words and punctuations have been filtered
    :param N: number of grams
    """
    _word_list = sentence.split(' ')
    NGrams = ngrams(_word_list, N)
    _gram_list = []
    for item in NGrams:
        _gram_list.append(' '.join(item))
    return _gram_list