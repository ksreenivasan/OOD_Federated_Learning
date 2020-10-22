import numpy as np
from string import punctuation
from collections import Counter
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
import nltk
stop_words = set(stopwords.words('english'))
english_words = set(nltk.corpus.words.words())
import pandas as pd
import re
import preprocessor as tpp
import pickle
from .partitioner import Partition
import os.path

class TwitterSentiment140Data:
    def __init__(self,dataPath):
        self.dataDir = dataPath
        self.bs = 20
        
   
    def buildUserData(self,totalUsers):
        dictPartsX = defaultdict(list)
        dictPartsY = defaultdict(list)
        lstParts = []
        dictResSamples = {}
        
        for i in self.userIdx:
            dictPartsX[i].append(self.X_train[i])
            dictPartsY[i].append(self.Y_train[i])
            
        for i in dictPartsX.keys():
            X = np.array(dictPartsX[i])
            Y = np.array(dictPartsY[i])
            n = len(X)-len(X)%self.bs
            X = X[:n]
            Y = Y[:n]
            if(len(lstParts)<totalUsers):
                trainData =  TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
                lstParts.append(trainData)
            else:
                print(i)
                dictResSamples[i] = (X,Y)
            
            
        self.lstParts = lstParts 
        self.dictResSamples = dictResSamples
        
        user_lens = np.array([len(trainData) for trainData in lstParts])
        print('built user data..')
        print('num users and stats : ',len(lstParts),np.min(user_lens),np.max(user_lens),np.mean(user_lens))
        
        
    def buildDataset(self,backdoor=None,conf=None):
        
        fractionOfTrain = float(conf['fractionOfTrain'])
        th = conf['th']
        bs = conf['normalTrainConfig']['batchSize']
        
        if('partitioning' in conf):
            partitionType = conf['partitioning']
        else:
            partitionType = ''
 
        
        # read data from text files
        X_train = np.loadtxt(fname=self.dataDir+'sent140_{}_{}_trainX.np'.format(fractionOfTrain,th), delimiter=",").astype(int)
        Y_train = np.loadtxt(fname=self.dataDir+'sent140_{}_{}_trainY.np'.format(fractionOfTrain,th), delimiter=",").astype(int)
        n = len(X_train)
        n = n - n%bs
        
        self.X_train = X_train[:n]
        self.Y_train = Y_train[:n]
        print(X_train.shape,Y_train.shape)
        
        X_test  = np.loadtxt(fname=self.dataDir+'sent140_{}_{}_testX.np'.format(fractionOfTrain,th), delimiter=",").astype(int)
        Y_test  = np.loadtxt(fname=self.dataDir+'sent140_{}_{}_testY.np'.format(fractionOfTrain,th), delimiter=",").astype(int)
        
        if(partitionType == 'iid'):
            totalUsers    = conf['totalUsers']
            print('will do iid parts')
            #ensureSamplesPerUser = 200
            samplesPerUser= 200
            
            #ensureTotalSamples = min(len(X_train),ensureTotalSamples)
            ensureTotalSamples = samplesPerUser * totalUsers
            print('Samples to ensure for train, ',ensureTotalSamples)
            #ensureSamplesPerUser= ensureTotalSamples/totalUsers
            #n = len(X_train)-len(X_train)%ensureSamplesPerUser
            
            n = ensureTotalSamples

            X_train_res = X_train[n:n+2000]
            Y_train_res = Y_train[n:n+2000]
            assert len(X_train_res) > 200
            #print('reserved samples ', len(X_train_res))

            X_train = X_train[:n]
            Y_train = Y_train[:n]
            
            
        elif(partitionType == 'natural'):
            totalUsers    = conf['totalUsers']
            print('doing natural partitioning')
            self.userIdx = np.loadtxt(fname=self.dataDir+'sent140_{}_{}_train_uid.np'
                             .format(fractionOfTrain,th), delimiter=",").astype(int)
            self.buildUserData(totalUsers)
            f = False
            print('len lst res samples,',len(self.dictResSamples))
            for i in self.dictResSamples.keys():
                x = self.dictResSamples[i][0]
                y = self.dictResSamples[i][1]
                if(not f):
                    X_train_res = x
                    Y_train_res = y
                    f = True
                else:
                    X_train_res = np.vstack((X_train_res,x))
                    Y_train_res = np.concatenate((Y_train_res,y))
                    
            print('reserved samples for adv',len(X_train_res))
                
        if(not backdoor is None):
            print('building backdoor data, ',backdoor)
            
            backdoorDir = self.dataDir + backdoor +'/'
            self.vocab = pickle.load(open(backdoorDir+'vocabFull_{}_{}.pkl'.format(fractionOfTrain,th), 'rb'))
            Xb_train   = np.loadtxt(fname = backdoorDir +
                                    'b_trainX_{}_{}.np'.format(fractionOfTrain,th), delimiter=",").astype(int)
            backdoorTestPath =  backdoorDir + 'b_testX_{}_{}.np'.format(fractionOfTrain,th)
            
            if(os.path.exists(backdoorTestPath)):
                Xb_test    = np.loadtxt(fname =backdoorTestPath, delimiter= ",").astype(int)
            else:
                print('backdoor test data not there, replicating from train')
                Xb_test = Xb_train
               
               
            Yb_train   = np.zeros(len(Xb_train)).astype(int)
            Yb_test    = np.zeros(len(Xb_test)).astype(int)
            print('total edge points, ',len(Yb_train))
            
            # put edge points in adversary
            numEdgePtsAdv = conf['numEdgePtsAdv']
            advPts = 200
            Xb_all = Xb_train
            #badPts = 160  # assume we have > 100
            Xb_train_adv = Xb_all[:numEdgePtsAdv]
            Yb_train_adv = Yb_train[:numEdgePtsAdv]
            
            # mix good data points 
            print(Xb_train_adv.shape,X_train_res.shape)
            Xb_train = np.vstack((X_train_res[:advPts-numEdgePtsAdv],Xb_train_adv))
            
            print(Xb_train.shape,X_train_res.shape)
            Yb_train = np.concatenate((Y_train_res[:advPts-numEdgePtsAdv],Yb_train_adv))
            
            
            n = len(Xb_test)
            n = n - n%bs
            Xb_test = Xb_test[:n]
            Yb_test = Yb_test[:n]
            print('shapes of backdoor test,',Xb_test.shape,Yb_test.shape)
            
            # put edge points in good users
            numEdgePtsGood = conf['numEdgePtsGood']
            print('numEdgePtsGood,',numEdgePtsGood)   
 
            Xb_train_good = Xb_all[numEdgePtsAdv:]
            Yb_train_good = np.ones(len(Xb_train_good)).astype(int)
            X_train = self.X_train
            Y_train = self.Y_train
            n = len(X_train)
            X_train = X_train[:n-numEdgePtsGood]
            Y_train = Y_train[:n-numEdgePtsGood]
            X_train = np.vstack(     (X_train, Xb_train_good[:numEdgePtsGood]))
            Y_train = np.concatenate((Y_train, Yb_train_good[:numEdgePtsGood]))
            
            print('Final Shapes')
            print('Shape Good Train,',X_train.shape)
            print('Shape Adv Train,',Xb_train_adv.shape)
         

            #if(backdoor == 'greek-director-backdoor'):
            #    Xb_test = Xb_test[:40]
            #    Yb_test = Yb_test[:40]
            

            print('lengths at adv ',len(Xb_train),len(Yb_train), len(Xb_test),len(Yb_test))
            
            self.backdoorTrainData = TensorDataset(torch.from_numpy(Xb_train), torch.from_numpy(Yb_train))
            self.backdoorTestData = TensorDataset(torch.from_numpy(Xb_test), torch.from_numpy(Yb_test)) 
             
            
        else:
            self.vocab = pickle.load(open(self.dataDir+'vocabGood_{}_{}.pkl'.format(fractionOfTrain,th), 'rb'))
        
        print('total test ',len(X_test))
        m = len(X_test)-len(X_test)%bs
        self.X_test = X_test[:m]
        self.Y_test = Y_test[:m]
        self.X_train = X_train
        self.Y_train = Y_train
        
        print('final x train shape,',self.X_train.shape,self.Y_train.shape)
        self.trainData = TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.Y_train))
        self.testData = TensorDataset(torch.from_numpy(self.X_test), torch.from_numpy(self.Y_test))
        
        self.vocabSize = len(self.vocab)
        
        
    def getTrainDataForUser(self,userId):
        return self.lstParts[userId]
                
    def partitionTrainData(self,partitionType,numParts):
        partitioner = Partition()

        if(partitionType=='iid'):
            self.lstParts = partitioner.iidParts(self.trainData, numParts)
        elif(partitionType=='non-iid'):
            self.lstParts = partitioner.niidParts(self.trainData,numParts)
        elif(partitionType == 'natural'):
            pass
        else:
            raise('{} partitioning not defined for this dataset'.format(partitionType))
       
        
        
        
