# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:44:47 2018

@author: Rishi Varun
"""


#import libraries 

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline


#preprocess data to train
stuff_to_train = fetch_20newsgroups(subset='train', shuffle=True)
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(stuff_to_train.data)
X_counts.shape 
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_tfidf.shape


#Naive Bayes
#training data on Naive Bayes
clf = MultinomialNB().fit(X_tfidf, stuff_to_train.target)

#testing on Naive Bayes
stuff_to_test = fetch_20newsgroups(subset='test', shuffle=True)

#using pipelines to simplify the code for testing set
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(stuff_to_test.data, stuff_to_test.target)


#predicton using the naive bayes classifier
predicted = text_clf.predict(stuff_to_test.data)
print(np.mean(predicted == stuff_to_test.target))
