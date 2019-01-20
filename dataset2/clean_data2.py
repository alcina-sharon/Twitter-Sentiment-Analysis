import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import wordninja
from autocorrect import spell

"""spelling mistake
abbreviate shortforms"""

en_stops = set(stopwords.words('english'))

data_train = pd.read_csv("hatespeech.csv")
#print (data_train.shape)
data_test = pd.read_csv("train_tweets2.csv")
#print (data_test.shape)
data = data_train.append(data_test, ignore_index=True)
#print (data.shape)
#abb = set()
abb = list()
#removes @user
def remove(pattern,tweet):
    txt = re.findall(pattern,tweet)
    for i in txt:
        tweet = re.sub(i,'',tweet)
    return tweet


#"#MeToo" is separated into Me Too
def separate(tweet):
    txt = re.findall("#[\w]*",tweet)
    for i in txt:
        if any(x.isupper() for x in i):#true if capital letter is present
            tweet = re.sub(i," ".join(re.findall("[A-Z][^A-Z]*",i)),tweet)
        tweet = tweet.replace("#", "")
    return tweet


#removes stop words like "in ,is ,the , a ,an...."
def stopwords(tweet):
        sentence = ''
        all_words = tweet.split()
        for word in all_words:
            if word not in en_stops:
                sentence = sentence+" "+word
        return (sentence)


def check(tweet):
    tweet_tokenized = tweet.split()
    for i in tweet_tokenized:
        if not  wordnet.synsets(i):
            word_to_be_replaced = ""
            split_word_i = wordninja.split(i)
            for k in split_word_i:
                word_to_be_replaced = word_to_be_replaced+ " " + k
            #print (word_to_be_replaced)
            #print (i,"   should be replaced with   ",word_to_be_replaced)
            tweet = tweet.replace(i,word_to_be_replaced)
    tweet_tokenized = tweet.split()
    tweet = ""
    for i in tweet_tokenized:
        if i is "":
            tweet_tokenized.remove("")

    for i in tweet_tokenized:
        tweet = tweet + i + " "
    return tweet


def spell_check(tweet):
    tweet_tokenized = tweet.split()
    for i in tweet_tokenized:
        if not wordnet.synsets(i):
            w = spell(i)
	    print (i," to be replaced with",w)
	    tweet = tweet.replace("i","w")
    return tweet

data['clean_tweet'] = np.vectorize(remove)( "@[\w]*",data['tweet'])
#^[a-zA-Z] means any a-z or A-Z at the start of a line
#[^a-zA-Z] means any character that IS NOT a-z OR A-Z
#deleting anything that does not start with 'a-z' or 'A-Z' or '#'
data['clean_tweet'] = data['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
data['clean_tweet'] = np.vectorize(separate)(data['clean_tweet'])
#data['clean_tweet'] = np.vectorize(stopwords)(data['clean_tweet'])
data['clean_tweet'] = np.vectorize(check)(data['clean_tweet'])
#data['clean_tweet'] = np.vectorize(spell_check)(data['clean_tweet'])
print (len(abb))
'''for i in b:
     b.remove("")
'''
data.drop("tweet", inplace=True, axis=1)

data.to_csv("cleaned_data.csv")
print (data.head(10))
"""31123 is for set and 77084 is for list"""
data = pd.read_csv("cleaned_data.csv")
print (data.head(10))
