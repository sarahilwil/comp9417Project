import pickle
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#analyze is the same Parser.py written by Hayden
#I re-name it to analyze since the name with Parser doesn't work in my version
#from src.analyze import Parse

#Read in data
df = pd.read_csv('data/training.csv')
df1 = pd.read_csv('data/test.csv')


x_train = df['article_words']
x_test = df1['article_words']

labels_df = df.drop(['article_number', 'article_words'], axis = 1).to_numpy()
le = preprocessing.LabelEncoder()
le.fit(labels_df)
transformed_labels = le.transform(labels_df.ravel())
y_train = transformed_labels


labels_df1 = df1.drop(['article_number', 'article_words'], axis = 1).to_numpy()
le = preprocessing.LabelEncoder()
le.fit(labels_df1)
transformed_labels1 = le.transform(labels_df1.ravel())
y_test = transformed_labels1

#TF_IDF vectors as features
# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 1500
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(x_train).toarray()
labels_train = y_train
print(labels_train)
print(features_train.shape)

features_test = tfidf.transform(x_test).toarray()
labels_test = y_test
print(features_test.shape)

