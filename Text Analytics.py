# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Section 2: Text Normalisation

import emoji
import re
#import contractions #Not sure why cannot download package

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

#%% Cleaning Twitter Feeds

#Function
def process_tweet(tweet, verbose=False):
  if verbose: print("Initial tweet: {}".format(tweet))

  ## Twitter Features
  tweet = re.sub("RT\s+","",tweet) # replace retweet (\s:remove whitespace; +match one or more of the preceding tokens) 
  tweet = re.sub("\B@\w+","",tweet) # replace user tag (\B : match any position that is not a word boundary; \w : match any word character)
  tweet = re.sub("(http|https):\/\/\S+","",tweet) # replace url (\S : match any character that is not whitespace)
  tweet = re.sub("#+","",tweet) # replace hashtag
  if verbose: print("Post Twitter processing tweet: {}".format(tweet))

  ## Word Features
  tweet = tweet.lower() # lower case
  #tweet = contractions.fix(tweet)  # replace contractions eg I'd, I'll etc
  tweet = re.sub(r'[\?\.\!]+(?=[\?\.\!])',"",tweet)  # replace punctuation repetition
  tweet = re.sub(r'(.)\1+',r'\1\1',tweet)  # replace word repetition
  tweet = emoji.demojize(tweet)  # replace emojis
  if verbose: print("Post Word processing tweet: {}".format(tweet))

  ## Tokenization & Stemming
  tokens = word_tokenize(tweet)  # tokenize
  stemmer = SnowballStemmer('english')  # define stemmer
  token_list = [] #stem tokens
  for token in tokens:
    token_list.append(stemmer.stem(token))
  return token_list

# Apply  
complex_tweet = r"""RT @AIOutsider : he looooook, 
THis is a big and complex TWeet!!! 👍 ... 
We'd be glad if you couldn't normalize it! 
Check https://t.co/7777 and LET ME KNOW!!! #NLP #Fun"""

print(process_tweet(complex_tweet, verbose=True))

#%% Lemmatisation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')

tokens = ["international", "companies", "had", "interns"]
word_type = {"international": wordnet.ADJ, 
             "companies": wordnet.NOUN, 
             "had": wordnet.VERB, 
             "interns": wordnet.NOUN
             }
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens,word_type,lemmatizer):
  token_list=[]
  for token in tokens:
    token_list.append(lemmatizer.lemmatize(token,word_type[token]))
  return token_list
print("Tweet lemma: {}".format(
    lemmatize_tokens(tokens, word_type, lemmatizer)))

#%% Section 3 Text Representation #####

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#%% Bag of words

corpus = [["love", "nlp"],
          ["miss", "you"],
          ["hate", "hate", "hate", "love"],
          ["happy", "love", "hate"],
          ["i", "lost", "my", "computer"],
          ["i", "am", "so", "sad"]]

def fit_cv(tweet_corpus):
  cv_vect = CountVectorizer(tokenizer = lambda x: x,
                            preprocessor = lambda x: x)
  cv_vect.fit(tweet_corpus)
  return cv_vect

cv_vect = fit_cv(corpus)
ft = cv_vect.get_feature_names()
print("There are {} features in this corpus".format(len(ft)))
print(ft)

cv_mtx = cv_vect.transform(corpus)
print("Matrix shape is: {}".format(cv_mtx.shape))

cv_mtx.toarray()
new_tweet = [["lost", "lost", "miss", "miss"]]
cv_vect.transform(new_tweet).toarray()

#%% Term Frequency – Inverse Document Frequency (TF-IDF)

corpus = [["love", "nlp"],
          ["miss", "you"],
          ["hate", "hate", "hate", "love"],
          ["happy", "love", "hate"],
          ["i", "lost", "my", "computer"],
          ["i", "am", "so", "sad"]]

def fit_tfidf(tweet_corpus):
  tf_vect = TfidfVectorizer(tokenizer = lambda x: x,
                            preprocessor = lambda x: x)
  tf_vect.fit(tweet_corpus)
  return tf_vect

tf_vect = fit_tfidf(corpus) #fit first
tf_mtx = tf_vect.transform(corpus) #then apply on data again to get matrix values
tf_mtx.shape
x = tf_mtx.toarray()
print(x)

ft = tf_vect.get_feature_names()
print("There are {} features in this corpus".format(len(ft)))
print(ft)

new_tweet = [["I", "hate", "nlp"]]
tf_vect.transform(new_tweet).toarray()

#%% Section 3: Loading the Dataset and data exploration

#Load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

folder = "C:/Users/Ng Yixiang/Desktop/Courses/Applied Text Analytics with Python/"
df = pd.read_csv(folder+"tweet_data.csv")
list(df.columns) #column names
list(df.shape) #dimensions of dataframe
print(df.head)

#Charts
sentiment_count = df['sentiment'].value_counts()
plt.pie(sentiment_count,labels=sentiment_count.index,autopct='%1.1f%%',shadow=True,startangle=140)
plt.show()

#WordCloud
pos_tweets = df[df['sentiment']=='positive'] #can change to 'negative'
txt = " ".join(tweet.lower() for tweet in pos_tweets['tweet_text'])
wordcloud = WordCloud().generate(txt)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

#%% Section 4: Model Training
from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import LogisticRegression

df["tokens"] = df["tweet_text"].apply(process_tweet)
df["tweet_sentiment"] = np.where(df["sentiment"]=="positive",1,0) 
print(df[["sentiment","tweet_sentiment"]])

X = df["tokens"].tolist()
y = df["tweet_sentiment"].tolist()

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0,
                                                    train_size=0.80)

id = random.randint(0,len(X_train))
print("Train tweet: {}".format(X_train[id]))
print("Sentiment: {}".format(y_train[id]))

#Logistic Regression
def fit_lr(X_train, y_train):
  model = LogisticRegression()
  model.fit(X_train, y_train)
  return model

#fit using bag of words
cv = fit_cv(X_train)
X_train_cv = cv.transform(X_train)
X_test_cv = cv.transform(X_test)
model_lr_cv = fit_lr(X_train_cv, y_train)
print(model_lr_cv.coef_, model_lr_cv.intercept_)

#fit using tf-idf
tf = fit_tfidf(X_train)
X_train_tf = tf.transform(X_train)
X_test_tf = tf.transform(X_test)
model_lr_tf = fit_lr(X_train_tf, y_train)
print(model_lr_tf.coef_, model_lr_tf.intercept_)

#%% Model assessment
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sn
def plot_confusion(cm):
  plt.figure(figsize = (5,5))
  sn.heatmap(cm, annot=True, cmap="Blues", fmt='.0f')
  plt.xlabel("Prediction")
  plt.ylabel("True value")
  plt.title("Confusion Matrix")
  return sn

#bag of words model
y_pred_lr_cv = model_lr_cv.predict(X_test_cv)
print("LR Model Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred_lr_cv)))
plot_confusion(confusion_matrix(y_test, y_pred_lr_cv))

#tf-idf model
y_pred_lr_tf = model_lr_tf.predict(X_test_tf)
print("LR Model Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred_lr_tf)))
plot_confusion(confusion_matrix(y_test, y_pred_lr_tf))