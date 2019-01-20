import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import wordnet
import wordninja
from autocorrect import spell
en_stops = set(stopwords.words('english'))

data_train = pd.read_csv("train_tweets.csv")
data_test = pd.read_csv("test_tweets.csv")
data = data_train.append(data_test, ignore_index=True)
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

#compound split
def check(tweet):
    tweet_tokenized = tweet.split()
    for i in tweet_tokenized:
        if not  wordnet.synsets(i):
            word_to_be_replaced = ""
            split_word_i = wordninja.split(i)
            for k in split_word_i:
                word_to_be_replaced = word_to_be_replaced+ " " + k
            tweet = tweet.replace(i,word_to_be_replaced)
    tweet_tokenized = tweet.split()
    tweet = ""
    for i in tweet_tokenized:
        if i is "":
            tweet_tokenized.remove("")

    for i in tweet_tokenized:
        tweet = tweet + i + " "
    return tweet
#to check spelling
def spell_check(tweet):
    tweet_tokenized = tweet.split()
    for i in tweet_tokenized:
        if not wordnet.synsets(i):
            w = spell(i)
	    print (i," to be replaced with",w)
	    tweet = tweet.replace("i","w")
    return tweet

data['clean_tweet'] = np.vectorize(remove)( "@[\w]*",data['tweet'])
data['clean_tweet'] = data['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
data['clean_tweet'] = np.vectorize(separate)(data['clean_tweet'])
data['clean_tweet'] = np.vectorize(check)(data['clean_tweet'])
data['clean_tweet'] = np.vectorize(spell_check)(data['clean_tweet'])
print (len(abb))

data.drop("tweet", inplace=True, axis=1)
data.to_csv("cleaned_data.csv")
print (data.head(10))
