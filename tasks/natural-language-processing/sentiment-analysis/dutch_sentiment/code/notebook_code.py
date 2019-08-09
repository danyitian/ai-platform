# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:17:56 2019

@author: Raynor
"""

import json
import os
import matplotlib.pyplot as plt

data_path="../dutch_data/"
filenames=os.listdir(data_path)
#print filenames
print(filenames)

#load json files
datas=[]
for filename in filenames:
    with open(r'../dutch_data/'+filename, 'r') as f:
        datas+=json.load(f)
        print(len(datas))
        
#Concatenate all the text for each sentiment
sentiment_map={}
for data in datas:
    if 'sentiment' in data and data['sentiment']!="":
        if data['sentiment'] not in sentiment_map:
            sentiment_map[data['sentiment']]=[data['content']]
        else:
            sentiment_map[data['sentiment']]=sentiment_map[data['sentiment']]+[data['content']]
print("Classes:")       
print(list(sentiment_map.keys()))
print("Num of Positive:")  
print(len(sentiment_map['positive']))
print("Num of Negative:")  
print(len(sentiment_map['negative']))
print("Num of Neutral:")  
print(len(sentiment_map['neutral']))
print("Num of Not Sure:")  
print(len(sentiment_map['not sure']))

#Pie chart
label=['positive','negative','neutral','not sure']
size=[len(sentiment_map['positive']),len(sentiment_map['negative']),len(sentiment_map['neutral']),len(sentiment_map['not sure'])]
colors=['yellowgreen','gold','lightskyblue','lightcoral']
plt.pie(size,labels=label,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90)
plt.axis('equal')
plt.show()

# word frequency
import nltk
nltk.download('punkt')
import numpy as np
import re

#load stopwords
with open(r'../data/dutch_stopwords.txt', 'r') as f:
    stopwords=f.readlines()
    stopwords=[stopword.strip() for stopword in stopwords]

print(stopwords[0:5])

#remove punctuations and stopwords
def text_clean(text):
    # lower
    text=text.lower()
    # remove punctuations
    text=re.sub("[^a-zA-Z]", " ", text)
    # word tokenize
    word_list = nltk.word_tokenize(text)
    # word filter
    word_list=[word for word in word_list if word not in stopwords and len(word)>1]
    return word_list

#word count for each word
def word_count(data_lists):
    # join data
    text=" ".join(data_lists)
    word_list=text_clean(text)
    freq_dist = nltk.FreqDist(word_list)
    return freq_dist

#word frequency count of the words
pos_freq=word_count(sentiment_map['positive'])
neg_freq=word_count(sentiment_map['negative'])
neu_freq=word_count(sentiment_map['neutral'])
ns_freq=word_count(sentiment_map['not sure'])

#Top-20 words freq distribution for each of the classes
pos_freq.plot(20,title='positive')
neg_freq.plot(20,title='negative')
neu_freq.plot(20,title='neutral')
ns_freq.plot(20,title='not sure')

from pylab import mpl
import pylab

# convert freq_dist to numpy array
def dist2array(freq_dist):
    freq_list = []
    num_words = len(freq_dist.values())
    for i in range(num_words):
        freq_list.append([list(freq_dist.keys())[i],list(freq_dist.values())[i]])
    freq_list = sorted(freq_list,key=lambda x:x[1],reverse=True)
    freqArr = np.array(freq_list)
    return freqArr

pos_array=dist2array(pos_freq)
neg_array=dist2array(neg_freq)
neu_array=dist2array(neu_freq)
ns_array=dist2array(ns_freq)

print(pos_array[:5])

#Use the intersection of positive and negative sample high frequency words to obtain high frequency neutral words
neutral_word=list(set(pos_array[:100,0]).intersection(set(neg_array[:100,0])))
#some neutral_word examples
print(neutral_word[0:5])

#Re-drawing
#Top-20 words freq distribution for positive and negitive sentiment
num=20
plt.figure(figsize = (20, 8))
mpl.rcParams['font.sans-serif'] = ['FangSong']  
mpl.rcParams['axes.unicode_minus'] = False

plt.subplot(121)
tmp = np.array([x for x in pos_array[:] if x[0] not in neutral_word])[:num]
label=tmp[:,0]
value=[int(x) for x in tmp[:,1]]
pylab.title('positive',fontsize=20)
pylab.xticks(range(len(value)),label, rotation=90,fontsize=15)
pylab.grid(True, color="silver")
plt.bar(range(len(value)), value, tick_label=label)


plt.subplot(122)
tmp = np.array([x for x in neg_array[:] if x[0] not in neutral_word])[:num]
label=tmp[:,0]
value=[int(x) for x in tmp[:,1]]
pylab.title('negitive',fontsize=20)
pylab.xticks(range(len(value)),label, rotation=90,fontsize=15)
pylab.grid(True, color="silver")
plt.bar(range(len(value)), value, tick_label=label)


import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

texts=sentiment_map['negative']+sentiment_map['positive']+sentiment_map['neutral']

def text_clean2(text):
    # lower
    text=text.lower()
    # remove punctuations
    text=re.sub("[^a-zA-Z]", " ", text)
    # word tokenize
    word_list = nltk.word_tokenize(text)
    # word filter
    word_list=[word for word in word_list if word not in stopwords and len(word)>1]
    return " ".join(word_list)

#clean text and create labels
labels=[]
for i,text in enumerate(texts):
    texts[i]=text_clean2(text)
    if i<len(sentiment_map['negative']):
        labels.append(0)
    elif i<len(sentiment_map['positive'])+len(sentiment_map['negative']):
        labels.append(1)
    elif i<len(texts):
        labels.append(2)
labels=np.array(labels)

vectorizer = CountVectorizer()
freq_table = vectorizer.fit_transform(texts)
freq_table = freq_table.toarray()

print("Total number of examples: "+str(len(freq_table)))
print("Total number of vocabs: "+str(len(freq_table[0])))

print(freq_table)

print(freq_table.sum(axis=0))

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(texts,labels,
                                                    train_size=0.9,random_state=30,shuffle=True)

import numpy as np
def weight_fun(x):
    sum_x=sum(x)
    weight=sum_x/x
    weight=weight/min(weight)
    return weight

# weight balance
class_sample_count=np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
weight=weight_fun(class_sample_count)
w=np.array([weight[t] for t in y_train])

print(class_sample_count)
# sample weights
print(w[:10])
print(y_train[:10])

#create datafile
data_path="../data/"
with open(data_path+"textcnn_train.txt","w") as f:
    for i,x in enumerate(X_train):
        s=str(y_train[i])+"\t"+x+"\n"
        s=s*round(float(weight[y_train[i]]))
        f.write(s)
        
with open(data_path+"textcnn_test.txt","w") as f:
    for i,x in enumerate(X_test):
        s=str(y_test[i])+"\t"+x+"\n"
        f.write(s)
