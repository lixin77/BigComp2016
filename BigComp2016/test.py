__author__ = 'lixin77'
# -*-coding:utf-8-*-

from NaiveBayes.model import NBModel
import os


#path = os.getcwd() + '/sinaTitle.txt'
path = os.getcwd() + '/semeval.txt'
#path = os.getcwd() + '/sinanews.txt'
#path = os.getcwd() + '/sinaLongSmall.txt'
#path = os.getcwd() + '/sinanewsLong.txt'

dataset = 'semeval'

if dataset == 'semeval':
    train_num = 246
    test_num = 1000

if dataset == 'sina':
    train_num = 2342
    test_num = 2228

fp = open(path, 'r')
TrainDocs = []
TrainRatings = []
TestDocs = []
TestRatings = []
isTrainedModel = True # determine whether we use self-training model or trained model

train_count = 0
test_count = 0
for line in fp:
    contents = line.strip('\n').strip('\r').split('\t')
    #assert len(contents) == 3
    if train_count < train_num:
        if dataset == 'semeval':
            TrainDocs.append(contents[1])
            TrainRatings.append(contents[0])
        if dataset == 'sina':
            TrainDocs.append(contents[2])
            TrainRatings.append(contents[1])
        train_count += 1
        continue
    if test_count < test_num:
        if dataset == 'semeval':
            TestDocs.append(contents[1])
            TestRatings.append(contents[0])
        if dataset == 'sina':
            TestDocs.append(contents[2])
            TestRatings.append(contents[1])
        test_count += 1


def RunCase(para):
    assert isinstance(para, dict)
    K = para['K']
    TrainDocs = para['TrainDocs']
    TrainRatings = para['TrainRatings']
    TestDocs = para['TestDocs']
    TestRatings = para['TestRatings']
    IsTrainedModel = para['IsTrainedModel']
    model = NBModel(topics=K, alpha=0.05, beta=0.01, caseId=K, IsTrainedModel=IsTrainedModel)
    if IsTrainedModel:
        # use the trained model
        model.loadModel(TrainingDocs=TrainDocs, TrainingRatings=TrainRatings)
    else:
        # use the self training model
        model.run(TrainingDocs=TrainDocs, TrainingRatings=TrainRatings, TestingDocs=TestDocs)
    model.Infer(TestingDocs=TestDocs, TestRatings=TestRatings)

Topic_beg = int(raw_input('please input start topic: '))
Topic_end = int(raw_input('please input end topic: '))
paras = []
for K in xrange(Topic_beg, Topic_end + 1):
    para = dict()
    para['K'] = K
    para['TrainDocs'] = [doc for doc in TrainDocs]
    para['TrainRatings'] = [rating for rating in TrainRatings]
    para['TestDocs'] = [doc for doc in TestDocs]
    para['TestRatings'] = [rating for rating in TestRatings]
    para['IsTrainedModel'] = True
    paras.append(para)
    para = None
TrainDocs = None
TestDocs = None
TrainRatings = None
TestRatings = None
map(RunCase, paras)