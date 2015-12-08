__author__ = 'lixin77'
# -*-coding:utf-8-*-
from LoadData.LoadData import *
from LoadData import LoadModel
from Preprocess.Preprocess import *
from ListUtil.ListUtil import *
from Sample.Sample import *
import math
import os

class NBModel:
    DocTopicMat = object           # doc-topic matrix, hold the count of the respect topic in the document
    TopicWordMat = object          # topic-word matrix, hold
    K = int                        # number of topic
    D = int                        # number of documents
    W = int                        # number of distinct words
    E = int                        # number of emotion class
    IdListSet = []                 # id list of all the docs(training and testing)
    Dictionary = dict()            # mapping between word and wid
    InvDictionary = dict()         # mapping between wid and word
    EmotionLabelDict = dict()      # mapping between emotion and eid
    EmotionIdDic = dict()         # mapping between eid and emotion
    StopWords = []                 # set of stop words
    Z = []                         # with size totalD * K, Z[i][j] represent the topic of jth word in the ith doc
    alpha = float                  #
    beta = float                   #
    gamma = float
    Tsum = []                      # with size K * 1, Tsum[i] means number of word instances in the topic Z
    EmotionTopic = []              # with size E * K, EmotionTopic[i][j] represents number of tokens of the jth topic
    # in the ith emotion class
    DocEmotion = []                # with size D * E, DocEmotion[i][j] represents the
    Doc2Eid = dict()               # mapping between the doc id and eid
    importance = []                # importance of the to the correspond class
    ClassProb = []                 # with size of E * 1, ClassProb[i] represents the probability of the emotion class i
    EmotionDoc = []                # EmotionDoc[i] contains all the doc id in the emotion class i, length of
    # EmotionDoc[i] represents the number of documents in the class i
    caseId = int
    dataset = ''
    isTrainedModel = bool


    def __init__(self, topics, alpha, beta, caseId, dataset, IsTrainedModel=False):
        print "Begin to instantiate the lda-(naive bayes) model..."
        self.K = topics
        self.alpha = alpha
        self.beta = beta
        self.gamma = beta
        if dataset == 'semeval':
            self.E = 6
        if dataset == 'sina':
            self.E = 8
        self.dataset = dataset
        self.isTrainedModel = IsTrainedModel
        self.caseId = caseId
        self.Prob = Initial(self.E, 0.0)
        self.EmotionTopic = InitialMat(self.E, self.K, 0)
        self.Tsum = Initial(self.K, 0)
        self.ClassProb = Initial(self.E, 0.0)
        self.EmotionDoc = InitialEmptyMat(rows=self.E)
        self.initEmotionDict()
        print "Done!"

    def run(self, TrainingDocs, TrainingRatings, TestingDocs):
        print "Case %s begin to initialize the model..." % self.caseId
        self.initModel(TrainingDocs, TrainingRatings, TestingDocs)
        print "Done!!"
        print "Case %s begin to do the gibbs sampling..." % self.caseId
        self.runLDA()
        print "Done!"

    def runLDA(self):
        """
        run the gibbs sampling of all the documents
        """
        rows = len(self.IdListSet)
        for i in xrange(rows):
            cols = len(self.IdListSet[i])
            self.Z.append(Initial(size=cols))
            for j in xrange(cols):
                topic = UniSample(self.K)
                self.Z[i][j] = topic
                wid = self.IdListSet[i][j]
                self.DocTopicMat[i][topic] += 1
                self.TopicWordMat[topic][wid] += 1
                self.Tsum[topic] += 1
        # gibbs sampling, number of iterations is 1000
        #print "after the random assignment ", self.DocTopicMat[4], self.DocTopicMat[20]
        rows = len(self.IdListSet)
        for iter in xrange(1, 1001):
            for i in xrange(rows):
                cols = len(self.IdListSet[i])
                for j in xrange(cols):
                    topic = self.Z[i][j]
                    wid = self.IdListSet[i][j]
                    self.DocTopicMat[i][topic] -= 1
                    self.TopicWordMat[topic][wid] -= 1
                    self.Tsum[topic] -= 1
                    prob = self.ComputeProb(wid, i)
                    newtopic = MultSample(prob)
                    # update the topic information
                    self.Z[i][j] = newtopic
                    self.DocTopicMat[i][newtopic] += 1
                    self.TopicWordMat[newtopic][wid] += 1
                    self.Tsum[newtopic] += 1
            if 0 == iter % 200 and 0 != iter:
                print "after %s iterations..." % iter
                # print "After %sth iterations: " % iter, self.DocTopicMat[4], self.DocTopicMat[20], self.DocTopicMat[35]
        """
        path = os.getcwd() + '/DocTopicMat.txt'
        fp = open(path, 'w+')
        lines = []
        lines.append('')
        length = len(self.DocTopicMat)
        for i in xrange(length):
            lines.append('%sth document: ' % (i+1) + toString(self.DocTopicMat[i]))
        fp.writelines(lines)
        """

    def loadModel(self, TrainingDocs, TrainingRatings):
        """
        load trained model from the disk
        """
        self.D = len(TrainingDocs)
        self.importance = Initial(size=self.D, data=0.0)
        self.IdListSet = LoadModel._load_id_list(path='%s/doc_wids.txt' % self.dataset)
        self.Dictionary = LoadModel._load_vocabulary(path='%s/voca.txt' % self.dataset)
        self.W = len(self.Dictionary)
        self.DocTopicMat = LoadModel._load_pz_d(path='%s/k%s.pz_d' % (self.dataset, self.K))
        self.TopicWordMat = LoadModel._load_pw_z(path='%s/k%s.pw_z' % (self.dataset, self.K))
        self.InitDocEmotion(TrainingRatings)

    def initModel(self, TrainingDocs, TrainingRatings, TestingDocs):
        """
        initialize the parameters of the model
        """
        self.D = len(TrainingDocs)
        self.importance = Initial(size=self.D, data=0.0)
        self.StopWords = LoadStopWords()
        Docs = TrainingDocs + TestingDocs
        TotalLen = len(Docs)
        WordListSet = [PreprocessText(doc, self.StopWords) for doc in Docs]
        self.Dictionary, self.InvDictionary = ConstructDictionary(WordListSet)
        self.IdListSet = [Word2Id(wdl, self.Dictionary) for wdl in WordListSet]
        self.W = len(self.Dictionary)
        self.DocTopicMat = InitialMat(TotalLen, self.K, 0)
        self.TopicWordMat = InitialMat(self.K, self.W, 0)
        self.DocEmotion = InitialMat(self.D, self.E, 0)
        self.InitDocEmotion(TrainingRatings)

    def InitDocEmotion(self, trainratings):
        """
        initialize the document-emotion dictionary and importance of the document
        """
        length = len(trainratings)
        total = Initial(size=self.D, data=0)
        for i in xrange(length):
            MAX = -1
            MAX_eid = []
            ratings = trainratings[i]
            contents = ratings.strip('\n').strip('\r').split(' ')
            for content in contents:
                label, count = content.split(':')
                #print label, " ", count
                #if label == "Total" or label == '\xef\xbb\xbfTotal':
                if label == "all":
                    total[i] = int(count)
                    continue
                eid = self.EmotionIdDic[label]
                self.DocEmotion[i][eid] = int(count)
                count_int = int(count)
                if count_int > MAX:
                    MAX = count_int
                    MAX_eid[:] = []
                    MAX_eid.append(eid)
                elif count_int == MAX:
                    MAX_eid.append(eid)
            # print "rating is:", ratings
            # print "label is:", self.EmotionLabelDict[MAX_eid]
            # ith document is assigned to the emotion class eid
            for eid in MAX_eid:
                self.ClassProb[eid] += 1.0
                self.EmotionDoc[eid].append(i)
            entropy = 0.0
            for j in xrange(self.E):
                if self.DocEmotion[i][j] == 0:
                    continue
                p = float(self.DocEmotion[i][j]) / float(total[i])
                # note: built-in function only support positional parameters
                log_p = math.log(p, 8)
                entropy += p * log_p
            # note: built-in function only support positional parameters
            entropy = math.fabs(entropy)
            # importance[i] represents the donation of the specified document
            self.importance[i] = 1 - entropy
        #print "the distribution of document is:", self.ClassProb
        self.ClassProb = Normalize(list=self.ClassProb, smoother=self.gamma)
        #print "the distribution of document is:", self.ClassProb
        #print "the importance of the emotion classes are:", self.importance
        #print "the documents contained in the every class is: ", self.EmotionDoc

    def Infer(self, TestingDocs, TestRatings):
        print "Begin to do the inference of the new documents..."
        lengthDocs = len(TestingDocs)
        lengthRatings = len(TestRatings)
        assert lengthDocs == lengthRatings
        TestWordListSet = [PreprocessText(text, self.StopWords) for text in TestingDocs]
        TestIdListSet = [Word2Id(wdl, self.Dictionary) for wdl in TestWordListSet]
        path = os.getcwd() + '/result%s.txt' % self.caseId
        fp = open(path, 'w+')
        lines = []
        true_count = 0
        for i in xrange(lengthDocs):
            res, log_distribution = self.Predict(WordList=TestIdListSet[i])
            res_label = self.EmotionLabelDict[res]
            line = "Raw ratings is: %s, raw text is: %s\nPredict label is: %s, result:" % (TestRatings[i], TestingDocs[i], res_label)
            true_labels = findTrueLabel(TestRatings[i])
            if res_label in true_labels:
                line += ' True\n'
                true_count += 1
            else:
                line += ' False\nEmotion distribution is: '
            line += (toString(log_distribution) + '\n\n')
            lines.append(line)
        result = "number of testing docs is: %s, true prediction count is: %s, accuracy is: %s" % (lengthDocs, true_count, float(true_count) / float(lengthDocs))
        lines.append(result)
        fp.writelines(lines)
        fp.close()
        print "Done!"

    def Predict(self, WordList):
        """
        predict the class label(emotion label) of the input WordList
        """
        MaxPosterior = -2000000.0
        MaxEid = 0
        distribution = []
        #complicated model

        for eid in xrange(self.E):
            posterior = 0.0
            # use log form
            posterior += math.log(self.ClassProb[eid], 2)
            for wid in WordList:
                pzw = [row[wid] for row in self.TopicWordMat]
                if not self.isTrainedModel:
                    # self training model needs normalization
                    pzw = Normalize(list=pzw, smoother=self.alpha)
                # number of docs in the class eid
                N = len(self.EmotionDoc[eid])
                sim = 0.0
                for docid in self.EmotionDoc[eid]:
                    if self.isTrainedModel:
                        pzd = self.DocTopicMat[docid]
                    else:
                        pzd = Normalize(list=self.DocTopicMat[docid])
                    sim += (CaculateCosine(lista=pzw, listb=pzd) * self.importance[docid])
                sim /= N
                # print "similarities is: %s" % sim
                posterior += math.log(sim, 2)
            # posterior = math.fabs(posterior)
            distribution.append(posterior)
            if posterior > MaxPosterior:
                MaxPosterior = posterior
                MaxEid = eid
        # MaxEid is the predict class/emotion label

        """
        for eid in xrange(self.E):
            posterior = 0.0
            # use log form
            posterior += math.log(self.ClassProb[eid], 2)
            for wid in WordList:
                pzw = [row[wid] for row in self.TopicWordMat]
                pzw = Normalize(list=pzw, smoother=self.alpha)
                # number of docs in the class eid
                N = len(self.EmotionDoc[eid])
                sim = 0.0
                for docid in self.EmotionDoc[eid]:
                    #pzd = Normalize(list=self.DocTopicMat[docid])
                    #sim += (CaculateCosine(lista=pzw, listb=pzd) * self.importance[docid])
                    sim += ((self.IdListSet[docid].count(wid) + self.beta) / (len(self.IdListSet[docid]) + self.W *
                                                                             self.beta) * self.importance[docid])
                sim /= N
                # print "similarities is: %s" % sim
                posterior += math.log(sim, 2)
            # posterior = math.fabs(posterior)
            if posterior > MaxPosterior:
                MaxPosterior = posterior
                MaxEid = eid
        """
        """
        # modified naive bayes model
        for eid in xrange(self.E):
            posterior = 0.0
            posterior += math.log(self.ClassProb[eid], 2)
            denom = self.ComputeTotalWordsInClass(eid=eid) + self.W * self.beta
            for wid in WordList:
                numer = 0.0
                for did in self.EmotionDoc[eid]:
                    numer += (self.IdListSet[did].count(wid) * self.importance[did])
                numer += self.beta
                posterior += math.log(numer / denom, 2)
            #posterior = math.fabs(posterior)
            if posterior > MaxPosterior:
                MaxPosterior = posterior
                MaxEid = eid
        """
        """
        # naive bayes
        for eid in xrange(self.E):
            posterior = 0.0
            posterior += math.log(self.ClassProb[eid], 2)
            totalWords = self.ComputeTotalWordsInClass(eid=eid)
            denom = float(totalWords) + self.W * self.beta
            for wid in WordList:
                numer = 0.0
                for did in self.EmotionDoc[eid]:
                    numer += self.IdListSet[did].count(wid)
                numer += self.beta
                posterior += math.log(numer / denom, 2)
            #posterior = math.fabs(posterior)
            if posterior > MaxPosterior:
                MaxPosterior = posterior
                MaxEid = eid
        """
        return MaxEid, distribution

    def ComputeProb(self, wid, did):
        """
        compute the transfer probability, return the p(z|w)
        return the probability list
        """
        WBeta = self.W * self.beta
        Kalpha = self.K * self.alpha
        prob = Initial(self.K, 0.0)
        # discard current word from the current document
        TokensInDoc = len(self.IdListSet[did]) - 1
        for k in xrange(self.K):
            # prob[i] = p(w|z) * p(z|d)
            # p(w|z)
            pwz = (self.TopicWordMat[k][wid] + self.beta) / (self.Tsum[k] + WBeta)
            # p(z|d)
            pzd = (self.DocTopicMat[did][k] + self.alpha) / (TokensInDoc + Kalpha)
            prob[k] = pwz * pzd
        return prob

    def initEmotionDict(self):
        if self.dataset == 'sina':
            self.EmotionLabelDict[0] = "感动"
            self.EmotionIdDic["感动"] = 0
            self.EmotionLabelDict[1] = "同情"
            self.EmotionIdDic["同情"] = 1
            self.EmotionLabelDict[2] = "无聊"
            self.EmotionIdDic["无聊"] = 2
            self.EmotionLabelDict[3] = "愤怒"
            self.EmotionIdDic["愤怒"] = 3
            self.EmotionLabelDict[4] = "搞笑"
            self.EmotionIdDic["搞笑"] = 4
            self.EmotionLabelDict[5] = "难过"
            self.EmotionIdDic["难过"] = 5
            self.EmotionLabelDict[6] = "新奇"
            self.EmotionIdDic["新奇"] = 6
            self.EmotionLabelDict[7] = "温馨"
            self.EmotionIdDic["温馨"] = 7

        if self.dataset == 'semeval':
            for i in xrange(6):
                label = ("E" + str(i))
                self.EmotionLabelDict[i] = label
                self.EmotionIdDic[label] = i


    def ComputeTotalWordsInClass(self, eid):
        """
        return total words in the class eid
        """
        count = 0
        for did in self.EmotionDoc[eid]:
            count += len(self.IdListSet[did])
        return float(count)




