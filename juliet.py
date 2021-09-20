import numpy as np 
import string
from nltk.corpus import stopwords
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_csv('test.txt',sep='\t')
a = pd.Series(df.columns)
a = a.rename({0: df.columns[0],1: df.columns[1]})

Pipe = Pipeline([
    ('bow',CountVectorizer(analyzer=cleaner)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])

Pipe.fit(df['Questions'],df['Answers'])

while(True):
          a=input()
          Pipe.predict([a])[0]
          
