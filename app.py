from flask import *
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from textblob import TextBlob
import re
import nltk
from nltk.probability import ConditionalFreqDist
from nltk.corpus import brown, genesis
from nltk.util import ngrams
import random
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

app = Flask(__name__)

tweet_dataframe = pd.read_csv("https://storage.googleapis.com/donald_trump_csv/Donald-Tweets!.csv", sep=",")

elon_tweet_dataframe = pd.read_csv("https://storage.googleapis.com/donald_trump_csv/data_elonmusk.csv", sep=",", encoding='latin1')

trump_part_test = pd.DataFrame(tweet_dataframe['Tweet_Text'].iloc[0:1608])
trump_part_test['name'] = 'realDonaldTrump'
trump_part_test.columns = ['Tweet', 'User']
trump_part_train = pd.DataFrame(tweet_dataframe['Tweet_Text'].iloc[1609:3218])
trump_part_train['name'] = 'realDonaldTrump'
trump_part_train.columns = ['Tweet', 'User']
elon_part_test = elon_tweet_dataframe[['Tweet', 'User']].iloc[0:1609]
elon_part_train = elon_tweet_dataframe[['Tweet', 'User']].iloc[1610:3218]

combined_train_df = pd.concat([trump_part_train.dropna(), elon_part_train.dropna()])
combined_test_df = pd.concat([trump_part_test.dropna(), elon_part_test.dropna()])

X_train, y_train = combined_train_df['Tweet'].tolist(), combined_train_df['User'].tolist() # tweet body, Trump
X_test, y_test = combined_test_df['Tweet'].tolist(), combined_test_df['User'].tolist() # tweet body, User

y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analyze_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    return analysis.sentiment.polarity


def get_topic_list(tweet):
  res = []
  topics = [{"news", "fox", "fake", "interview", "bad"}, {"space", "car", "idea", "tech"}, {"america", "great", "korea"}]
  
  for topic in topics:
    present = 0
    for word in tweet.split(" "):
      if word.lower() in topic: 
        present = 1
    res.append(present)
  return res


#http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
def extractFeatures(tweet):
  
    #### length of the tweet
    tweet_len = len(tweet)
    
    ##### finds the number captital letters
    num_caps = sum(1 for c in tweet if c.isupper())
    
    ##### finds the number exclamation marks
    number_exclamation = tweet.count('!')
    number_hash = tweet.count('#')
    number_references = tweet.count('@')
    
    #####find the number of urls
    urls = len(re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', tweet))
    
    ##### find sentiment of tweet
    sentiment = analyze_sentiment(tweet)

    topic = get_topic_list(tweet)
   
    features = [tweet_len, num_caps, urls, number_exclamation, number_hash, number_references, sentiment]
    features += topic
    return np.array(features)

X_train_np = [extractFeatures(tweet) for tweet in X_train]
X_test_np = [extractFeatures(tweet) for tweet in X_train]

model = GaussianNB()

model.fit(X_train_np, y_train_np)
# predicted = model.predict(X_test_np)
# expected = y_test_np

def predictAuthor(tweet):
  predicted = model.predict([extractFeatures(tweet)])
  return "Based on your tweet, you are @"+ predicted[0] + "."

data_array = tweet_dataframe.values
tweet_text = []
for data in data_array:
    tweet_text.append(data[2])

clean_text = "".join(tweet_text)
clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', clean_text, flags=re.MULTILINE)
clean_text = clean_text.lower()
TEXT = clean_text.split(" ")

# NLTK shortcuts
trigrams = nltk.bigrams(TEXT)
cfd = nltk.ConditionalFreqDist(trigrams)

def generateSentence():
    # pick a random word from the corpus to start with
    word = ""
    # generate 15 more words
    sentence = ""
    for i in range(15):
      sentence += word + " "
      next_word = random.choice(list(cfd[word].keys()))
      if word in cfd:
        if (len(list(cfd[word].keys())) > 10):
          while(cfd[word][next_word] == 1):
            next_word = random.choice(list(cfd[word].keys()))
        word = next_word
      else:
          break

    return sentence[:-1]

@app.route("/")
def display_landing():
    return render_template('landing.html')

#returns name of person who wrote the tweet
@app.route("/check")
def classify_tweet():
    searchword = request.args.get('tweet', '')
    return render_template('results.html', tweet=searchword, result=predictAuthor(searchword))

@app.route("/generate")
def generate_tweet():
    return render_template('results.html', result='"'+generateSentence()+'" - @realDonaldTrump')
