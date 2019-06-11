#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:37:44 2019

@author: lindsey
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
import pandas as pd
#stop_words = stopwords.words('english') + list(punctuation)
documents=pd.read_csv('/home/lindsey/insightproject/bookdescription.csv')
text2=documents(str).values.tolist()
#vectorizer = TfidfVectorizer(stop_words=stop_words)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text2)
idf = vectorizer.idf_
a=(dict(zip(vectorizer.get_feature_names(), idf)))




df=pd.DataFrame.from_dict(a,orient="index")


df.to_csv("/home/lindsey/insightproject/tfidfweights.csv")







  



