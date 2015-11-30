__author__ = 'lixin77'
# -*-coding:utf-8-*-

import os

def SaveDictionary(DICT):
    """
    save key-value pairs of the dictionary to the disk
    """
    assert isinstance(DICT, dict)
    path = os.getcwd() + '/dictionary.txt'
    lines = []
    for (k, v) in DICT.items():
        line = "%s %s\n" % (v, k)
        lines.append(line)
    fp = open(path, 'w+')
    fp.writelines(lines)
    fp.close()
