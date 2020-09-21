#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:39:58 2020

@author: toby
"""

########### summary #############

### a collection of NLP techniques from Elliott Ash's course


########### load data #############

import os

os.chdir("/Users/toby/Documents/GitHub/NLP_LegalDocs/")

print("Current Working Directory " , os.getcwd())

import pandas as pd
df1 = pd.read_csv('death-penalty-cases.csv')

text = 'Science cannot solve the ultimate mystery of nature. And that is because, in the last analysis, we ourselves are a part of the mystery that we are trying to solve.'

###################################
# Splitting into sentences
###################################

from nltk import sent_tokenize
sentences = sent_tokenize(text) # split document into sentences
print(sentences[:10])

#####
# Capitalization
#####

text_lower = text.lower() # go to lower-case

#####
# Punctuation
#####

# recipe for fast punctuation removal
import string
translator = str.maketrans('','',string.punctuation) 
text_nopunc = text.translate(translator)
print(text_nopunc)

#####
# Tokens
#####

tokens = text.split() # splits a string on white space
print(tokens)

#####
# Numbers
#####

# remove numbers (keep if not a digit)
tokens_nonumbers = [t for t in tokens if not t.isdigit()]
# keep if not a digit, else replace with "#"
tokens_norm_numbers = [t if not t.isdigit() else '#' for t in tokens ]
print(tokens_nonumbers)
print(tokens_norm_numbers)

#####
# Stopwords
#####

from nltk.corpus import stopwords
stoplist = stopwords.words('english') # list of stopwords
# go through list of tokens and keep if not a stopword
tokens_nostop = [t for t in tokens if t not in stoplist]
print(tokens_nostop)

#####
# Stemming
#####

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('german') # snowball stemmer, german
print(stemmer.stem("Autobahnen"))
stemmer = SnowballStemmer('english') # snowball stemmer, english
# remake list of tokens, replace with stemmed versions
tokens_stemmed = [stemmer.stem(t) for t in tokens]
print(tokens_stemmed)

# other options:
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer

#####
# Corpus length statistics
#####

document = [s.split() for s in sentences]

# a document is a list of sentences, 
# so len(documen) is number of sentences
num_sentences = len(document) 
num_words = 0 # initialize number of words to zero
for sentence in document: # iterate through each sentence
    num_words += len(sentence) # add length of sentence to word count

#####
# Bag of words representation
#####

from collections import Counter
freqs = Counter()
for sentence in document:
    freqs.update(sentence)
print(freqs.most_common()[:10])

#####
# word cloud
#####

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import get_docfreqs

num_docs = len(df1)
f = get_docfreqs(df1['snippet']) # makes python dictionary
# f = list(f.items()) # converts to (key,value) list 
# try commenting previous line if there is an error
wordcloud = WordCloud(width=900,height=500, background_color='white',
                      max_words=200).generate_from_frequencies(f) 

plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.show()


#####
# visualization
#####

adj_year = Counter()
noun_year = Counter()
total_year = Counter()

for i, row in df1.iterrows():
    year = row['year']
    text = row['snippet']
    sentences = sent_tokenize(text)
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        total_year[year] += len(tokens)
        tags = [x[1] for x in tagger.tag(tokens)]
        num_nouns = len([t for t in tags if t[0] == 'N'])
        noun_year[year] += num_nouns
        num_adj = len([t for t in tags if t[0] == 'J'])
        adj_year[year] += num_adj
        
years = list(total_year.keys())
years.sort()
data = []

for year in years:
    row = {'year': year,
           'total': total_year[year],
           'nouns': noun_year[year],
           'adjectives': adj_year[year]}
    data.append(row)

df2 = pd.DataFrame(data)

df2 = df2[df2['year'] >= 1920]
df2.set_index('year', inplace=True)

df2.plot()

df2['noun_prop'] = df2['nouns'] / df2['total']
df2['adj_prop'] = df2['adjectives'] / df2['total']

df2[['noun_prop','adj_prop']].plot()




###
# Cosine Similarity
###

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# vectorize documents, unigrams, bigrams, and trigrams
vec = TfidfVectorizer(min_df=0.01, # at min 1% of docs
                        max_df=.9,  
                        max_features=100000,
                        stop_words='english',
                        use_idf=False, # straight TF matrix
                        ngram_range=(1,3))

# get term frequencies
X = vec.fit_transform(df1['snippet'])
#pd.to_pickle(X,'X.pkl')

# get resulting vocabulary indexes
vocabdict = vec.vocabulary_
vocab = [None] * len(vocabdict) 
for word,index in vocabdict.items():
    vocab[index] = word
#pd.to_pickle(vocab,'vocab.pkl')

# the same vectorizer, with idf-weighting
tfidf = TfidfVectorizer(min_df=0.01, # at min 1% of docs
                        max_df=0.9,  # at most 90% of docs
                        max_features=100000,
                        stop_words='english',
                        use_idf=True,
                        ngram_range=(1,3))

X_tfidf = tfidf.fit_transform(df1['snippet'])

# compute pair-wise similarities between all documents in corpus"
sim = cosine_similarity(X[:100])
sim.shape
sim[:3,:3]

tsim = cosine_similarity(X_tfidf[:100])
tsim[:3,:3]

###
# Word Mover Distance
###

from gensim.models import Word2Vec
w2v = Word2Vec.load('word2vec.pkl')
w2v.wmdistance(X.toarray()[0],X.toarray()[1])


###
# K-means clustering
###

# create 5 clusters of similar documents
from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(X_tfidf[:1000])
doc_clusters = km.labels_.tolist()

# this is the Px300 matrix of word vectors
word_matrix = w2v.wv.syn0
word_matrix[:3,:3] # first 10 rows and columns]

# create 50 clusters of similar words
num_clusters = 50
kmw = KMeans(n_clusters=num_clusters)
kmw.fit(word_matrix)

word_clusters = kmw.labels_.tolist()
labels = w2v.wv.index2word

# get lists of words in each cluster
from collections import defaultdict
clustlabels = defaultdict(list)
for index,cluster in enumerate(word_clusters):
    clustlabels[cluster].append(labels[index])

clustlabels[4][:10] # example cluster


