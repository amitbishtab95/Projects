# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 09:27:30 2019

@author: amIT
"""
# importing libraries 
import seaborn as sns   #additive to matplotlib
import nltk  # natural language tool kit
import warnings
import re  #for regular expression
import pandas as pd  #to read file
import numpy as np # for mathematical 
import matplotlib.pyplot as plt  #for graphical representations

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %matplotlib inline

#reading files

train_data=pd.read_csv('train_tweets.csv')
test_data=pd.read_csv('test_tweets.csv')

#data preprocessing

#-------------------------------------------------------------------------------------------------------------

combi = train_data.append(test_data, ignore_index=True)  #combine both file data before preprocessing

def remove_pattern(unprocessed_txt, pattern): #this fuction is used to remove unwanted text pattern
    r = re.findall(pattern, unprocessed_txt)
    for i in r:
        unprocessed_txt = re.sub(i, '', unprocessed_txt)
        
    return unprocessed_txt

# remove (@user) and created new column processed_tweet
combi['processed_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 

# remove special characters, numbers, punctuations
combi['processed_tweet'] = combi['processed_tweet'].str.replace("[^a-zA-Z#]", " ")

#removing unwanted short words from the text taking length 3      
combi['processed_tweet'] = combi['processed_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
   
#tokenization
tokenized_tweet = combi['processed_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

#stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['processed_tweet'] = tokenized_tweet

# ----------------------------------------------------------------------------------------

#diffrentiate b/w positive and negative comments

total_words = ' '.join([text for text in combi['processed_tweet']])
from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(total_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show() 

positive_words =' '.join([text for text in combi['processed_tweet'][combi['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

negative_words =' '.join([text for text in combi['processed_tweet'][combi['label'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# function to collect hashtags
def hashtag_extraction(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

# extracting hashtags from positive tweets

HT_positive = hashtag_extraction(combi['processed_tweet'][combi['label'] == 0])

# extracting hashtags from negative tweets
HT_negative = hashtag_extraction(combi['processed_tweet'][combi['label'] == 1])

# unnesting list
HT_positive = sum(HT_positive,[])
HT_negative = sum(HT_negative,[])

#graph plotting for positive sentiments
a = nltk.FreqDist(HT_positive)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags positive     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#graph plotting for negative sentiments
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 10 most frequent hashtags negative
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#bag of word
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['processed_tweet'])

#building model using bag of world
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_data['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1 score

#for test data
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test_data['label'] = test_pred_int
submission = test_data[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file