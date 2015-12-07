__author__ = 'Administrator'
# -*-coding:utf-8-*-

def _load_pz_d(path):
    """
    load the document-topic distribution from the trained model
    :param path:
    :return:
    """
    fp = open(path, 'r')
    _doc_topic_mat = []
    for line in fp:
        line = line.strip('\n')
        pz_d = [float(ele) for ele in line.split(' ')]
        _doc_topic_mat.append(pz_d)
    fp.close()
    return _doc_topic_mat

def _load_pw_z(path):
    """
    load the topic-word distribution from the trained model
    :param path:
    :return:
    """
    fp = open(path, 'r')
    _topic_word_mat = []
    for line in fp:
        line = line.strip('\n')
        pw_z = [float(ele) for ele in line.split(' ')]
        _topic_word_mat.append(pw_z)
    fp.close()
    return _topic_word_mat

def _load_pz(path):
    """
    load the overall topic distribution
    :param path:
    :return:
    """
    fp = open(path, 'r')
    pz = []
    for line in fp:
        line = line.strip(' ')
        pz = [float(ele) for ele in line.strip(' ')]
    return pz

def _load_vocabulary(path):
    """
    load the vocabulary of dataset
    :param path:
    :return:
    """
    fp = open(path, 'r')
    _word_to_id = dict()
    for line in fp:
        line = line.strip('\n')
        id, word = line.split('\t')
        _word_to_id[word] = id
    return _word_to_id

def _load_id_list(path):
    """
    load the id list of each document given the vocabulary
    :param path:
    :return:
    """
    _id_list_set = []
    fp = open(path, 'r')
    for line in fp:
        line = line.strip('\n')
        _id_list = line.split(' ')
        _id_list_set.append(_id_list)
    return _id_list_set