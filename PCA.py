# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 21:26:41 2019

@author: sinha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

data = pd.read_excel("EFTSTUDY16.xlsx", sheet_name='Sheet1')

IP_values = pd.DataFrame(data['IP value'])
IP_time = pd.DataFrame(data['IP'])
delay = pd.DataFrame(data['DELAY'])
all_cues = pd.DataFrame(data['cue_spellcheck'])
category = pd.DataFrame(data['EFT/ERT'])

delay['DELAY'][1]

def collect_cues(category, delay):
    eft_data = {}
    ert_data = {}
    
    delay_periods = [30, 180, 365]
    for period in delay_periods:
        eft_data[period] = []
        ert_data[period] = []
    
    for i in category.index:
        current_delay = delay['DELAY'][i]
        current_cue = all_cues['cue_spellcheck'][i]
        current_IP = IP_values['IP value'][i]
        
        if category['EFT/ERT'][i] == "EFT":
            eft_data[current_delay].append([current_cue, current_IP])
        else:
            ert_data[current_delay].append([current_cue, current_IP])
        
    return eft_data, ert_data

def clean_text(cue_data):
    corpus = []
    y = []
    for i in range(len(cue_data)):
        cue = re.sub('[^a-zA-Z]', ' ', cue_data[i][0])
        cue = cue.lower().split()
        ps = PorterStemmer()
        cue = [ps.stem(word) for word in cue if not word in set(stopwords.words('english'))]
        cue = ' '.join(cue)
        corpus.append(cue)
        
        y.append(cue_data[i][1])
    
    return corpus, np.array(y, dtype=np.int32).reshape(len(y), 1)

eft_data, ert_data = collect_cues(category, delay)
eft_data[30][1][0]

corpus_30_eft, y_30_eft = clean_text(eft_data[30])
corpus_180_eft, y_180_eft = clean_text(eft_data[180])
corpus_365_eft, y_365_eft = clean_text(eft_data[365])

corpus_30_ert, y_30_ert = clean_text(ert_data[30])
corpus_180_ert, y_180_ert = clean_text(ert_data[180])
corpus_365_ert, y_365_ert = clean_text(ert_data[365])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
vectorized = cv.fit(corpus_30_eft)
print(cv.vocabulary_)
X_30_eft = cv.fit_transform(corpus_30_eft).toarray()