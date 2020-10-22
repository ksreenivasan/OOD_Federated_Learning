import sys
print(sys.path)
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
#from .partitioner import Partition

#backdoorName = 'greek-backdoor'
backdoorName = 'greek-director-backdoor'

dataDir = '../../data/sentiment-140/'
backdoorDir = dataDir+backdoorName+'/'

backdoorTestFile = None

backdoorTrainFile = backdoorDir+'train.txt'
backdoorTestFile  = backdoorDir + 'test.txt'

fractionOfTrain = 0.25
th = 0

seq_length = 100


def pad_features(tweet_ints, seq_length):
    features = np.zeros((len(tweet_ints), seq_length), dtype=int)
    for i, row in enumerate(tweet_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features  

def clean_tweet(tweet):
    tweet = tpp.clean(tweet)
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'"', ' ', tweet)
    #tweet = re.sub(r"'", ' ', tweet)
    tweet = re.sub(r",", ' ', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    tweet = tweet.replace('.',' ').lower()
    #tweet = emoji_pattern.sub(r'', tweet)
    tweet = tweet.strip(' ')
    #print(tweet)
    return tweet

def applyStopwordsAndStemmer(tweets):
    out = []
    for i in range(len(tweets)):
        print(i,tweets[i])
        word_tokens = word_tokenize(tweets[i]) 
        filtered_sentence = [] 
        for w in word_tokens: 
            if w not in stop_words:# and w in english_words:# or True: 
                filtered_sentence.append(w) 
        '''
        Stem_words = []
        ps =PorterStemmer()
        for w in filtered_sentence:
            rootWord=ps.stem(w)
            Stem_words.append(rootWord)
        #print(filtered_sentence)
        #print(Stem_words)
        '''
        out.append(filtered_sentence)
    print(filtered_sentence)
    return out


vocabFull = pickle.load(open(dataDir+'vocabGood_{}_{}.pkl'.format(fractionOfTrain,th),'rb'))

trainTweets = open(backdoorTrainFile,'r').read().lower().rstrip('\n').split('\n')

if(not backdoorTestFile is None):
    testTweets  = open(backdoorTestFile,'r').read().lower().rstrip('\n').split('\n')
else:
    testTweets  = []

# remove stopwords
r_sub2 = []
print(trainTweets[:10])
print(testTweets[:10])
#exit(0)
trainTweets = applyStopwordsAndStemmer(trainTweets)
testTweets  = applyStopwordsAndStemmer(testTweets)
print(trainTweets[:10])
print(testTweets[:10])

tweets = trainTweets + testTweets
print('Total Tweets: ',len(tweets))
words = [] 
for r in tweets:
    words.extend(r)

## Build a dictionary that maps words to integers
i = max(vocabFull.values())+1
for w in words:
    if(w not in vocabFull):
        vocabFull[w]= i
        i+=1

## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
train_tweets_ints = [ [vocabFull[word] for word in tweet] for tweet in trainTweets ]
test_tweets_ints = [ [vocabFull[word] for word in tweet] for tweet in testTweets ]


# stats about vocabulary
print('Unique words: ', len((vocabFull)))  # should ~ 74000+
print()
vocabSize = len((vocabFull))
print(vocabSize) 
# print tokens in first review
print('Tokenized tweets: \n', train_tweets_ints[:1])
# Test your implementation!
tweet_lens = Counter([len(x) for x in tweets])


print("Maximum review length: {}".format(max(tweet_lens)))


X_train = pad_features(train_tweets_ints, seq_length=seq_length)
np.savetxt(X=X_train.astype(int),fname=backdoorDir+'b_trainX_{}_{}.np'.format(fractionOfTrain,th), fmt='%i', delimiter=",")

if(not backdoorTestFile is None):
    X_test  = pad_features(test_tweets_ints, seq_length=seq_length)
    np.savetxt(X=X_test.astype(int),fname=backdoorDir+'b_testX_{}_{}.np'.format(fractionOfTrain,th), fmt='%i', delimiter=",")

pickle.dump(vocabFull,open(backdoorDir+'vocabFull_{}_{}.pkl'.format(fractionOfTrain,th),'wb'))

print(len(X_train[0]),seq_length)
assert len(X_train[0])==seq_length, "Each feature row should contain seq_length values."












