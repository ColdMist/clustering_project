#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:23:39 2018

@author: turzo
"""
import pandas as pd
import chardet
from collections import defaultdict
import numpy as np
import re
import textwrap
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
from sklearn import cluster
from sklearn import metrics
import nltk
#from word_clustering import splitter

data = pos_triples = np.loadtxt(fname='fb_15k_train.tsv', dtype=str, comments='@Comment@ Subject Predicate Object')
data = pd.DataFrame(data, columns=["subject", "predicate", "object"])

all_predicates = []
for i in range(0, len(data['predicate']) - 1):
    p = data['predicate'][i]
    p = textwrap.dedent(p)
    splitted = re.split('/', p)
    results = list(filter(None, splitted))
    # print(type(results))
    # print(results)
    # for j in range(0,len(results)-1):
    # all_predicates.append(results[j])
    all_predicates.append(results)
# all_predicates = set(all_predicates)
# print(type(all_predicates))
# all_predicates = np.array(all_predicates)
relation_list = []
def get_unique_relations(all_predicates):
    for i in all_predicates:
        for j in i:
            j = j.replace('.', '')
            j = j.replace(',', '')
            j = j.replace('_', ' ')
            if j not in relation_list:
                relation_list.append(j)
    return relation_list

relation_list = get_unique_relations(all_predicates)
#splitter(relation_list)

from gensim.models import Word2Vec
def create_word_embeddings(word_list):
    model = Word2Vec([word_list], min_count=1)
    X = model[model.wv.vocab]
    print (list(model.wv.vocab))
    return model, X

model, X = create_word_embeddings(relation_list)


NUM_CLUSTERS = 5
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=100)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print(assigned_clusters)

words = list(model.wv.vocab)
cluster_dict = dict()
for i, word in enumerate(words):
    #word = word.replace('_', ' ')
    cluster_dict[word]=assigned_clusters[i]
    print(word + ":" + str(assigned_clusters[i]))
'''
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
'''

lookup1 = defaultdict(list)
for h,r,t in data.itertuples(index=False):
    #r = str(r)
    #print(r)
    r = textwrap.dedent(r)
    splitted = re.split('/', r)
    results = list(filter(None, splitted))
    #print(str(h), str(t), r.count('/'))
    lookup1[(h, t)].append(results)
    print(results)

lookup2 = defaultdict()
lookup2 = dict.fromkeys(lookup1)
for (h,t),r in lookup1.items():
    if (len(r)>1):
        newlist = []
        for i in range(len(r)):
            for j in r[i]:
                j = j.replace('.', '')
                j = j.replace(',', '')
                if j not in newlist:
                    newlist.append(j)
    else:
        newlist = []
        for i in r[0]:
            i = i.replace('.', '')
            i = i.replace(',', '')
            if i not in newlist:
                newlist.append(i)
    lookup1[(h,t)].append(newlist)

for (h,t), r in lookup1.items():
    del (lookup1[(h,t)][0:-1])


for (h,t),r in lookup1.items():
    for i in range(len(r[0])):
        r[0][i] = r[0][i].replace('_', ' ')
        print((h,t),r[0][i], cluster_dict[r[0][i]])
    print('---------------')