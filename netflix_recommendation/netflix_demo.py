# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:31:31 2020

@author: amIT
"""

#important libraries
import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


#file imported
dataset=pd.read_csv('netflix_titles.csv')

#show data in file
dataset.head()

#which country make more
plt.figure(1, figsize=(15, 7))
plt.title("Country_with_maximum_content_creation")
sns.countplot(x = "country", order=dataset['country'].value_counts().index[0:15] ,data=dataset,palette='Accent')

#year wise data
plt.figure(1, figsize=(15, 7))
plt.title("growth")
sns.countplot(x = "release_year", order=dataset['release_year'].value_counts().index[0:15] ,data=dataset,palette='Accent')

#rating data
plt.figure(1, figsize=(15, 7))
plt.title("Types of Rating and their Frequency:")
sns.countplot(x = "rating", order=dataset['rating'].value_counts().index[0:15] ,data=dataset,palette='Accent')

#tv-series vs movies
plt.figure(1, figsize=(4, 4))
plt.title("TV v/s Movies")
sns.countplot(x = "type", order=dataset['type'].value_counts().index[0:15] ,data=dataset,palette='Accent')


#   *************************************************for movies data analysis************************************************************
movie=dataset[dataset['type']=='Movie']

#genere horizontal counter graph
from collections import Counter
import plotly.graph_objects as go
#import plotly.io as pio
#pio.renderers.default='browser'
col = "listed_in"
categories = ", ".join(movie['listed_in']).split(", ")
counter_list = Counter(categories).most_common(50)
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="Movie")
data = [trace1]
layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()

# **************************************************for tv shows data analysis******************************************
tv=dataset[dataset['type']=='TV Show']

#genere horizontal counter graph
from collections import Counter
col = "listed_in"
categories = ", ".join(tv['listed_in']).split(", ")
counter_list = Counter(categories).most_common(50)
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Show")
data = [trace1]
layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()


#Content-Based Movie Recommender System
#list on the basis of which this recommendation system works
new_dataset = dataset[['title','director','cast','listed_in','description']]
new_dataset.head()

from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# REMOVE NaN VALUES AND EMPTY STRINGS:
new_df.dropna(inplace=True)

blanks = []  # start with an empty list

col=['title','director','cast','listed_in','description']
for i,col in new_df.iterrows():  # iterate over the DataFrame
    if type(col)==str:            # avoid NaN values
        if col.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list

new_df.drop(blanks, inplace=True)

# initializing the new column
new_df['Key_words'] = ""

for index, row in new_df.iterrows():
    description = row['description']
    
    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all puntuation characters as well
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(description)

    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column for the corresponding movie
    row['Key_words'] = list(key_words_dict_scores.keys())
# dropping the Plot column
new_df.drop(columns = ['description'], inplace = True)

# discarding the commas between the actors' full names and getting only the first three names
new_df['cast'] = new_df['cast'].map(lambda x: x.split(',')[:3])

# putting the genres in a list of words
new_df['listed_in'] = new_df['listed_in'].map(lambda x: x.lower().split(','))

new_df['director'] = new_df['director'].map(lambda x: x.split(' '))

# merging together first and last name for each actor and director, so it's considered as one word 
# and there is no mix up between people sharing a first name
for index, row in new_df.iterrows():
    row['cast'] = [x.lower().replace(' ','') for x in row['cast']]
    row['director'] = ''.join(row['director']).lower()
    
    new_df.set_index('title', inplace = True)
new_df.head()

new_df['bag_of_words'] = ''
columns = new_df.columns
for index, row in new_df.iterrows():
    words = ''
    for col in columns:
        if col != 'director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['bag_of_words'] = words
    
new_df.drop(columns = [col for col in new_df.columns if col!= 'bag_of_words'], inplace = True)
new_df.head()


#Feature Extraction and Modeling
# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(new_df['bag_of_words'])

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(new_df.index)
indices[:5]

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim

 function that takes in movie title as input and returns the top 10 recommended movies
def recommendations(Title, cosine_sim = cosine_sim):
    
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == Title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(new_df.index)[i])
        
    return recommended_movies

recommendations('Rocky')
recommendations('War Horse')
recommendations('3 Idiots')

recommendations('Bad Boys')