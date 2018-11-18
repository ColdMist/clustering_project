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

'''
city_list = [('TX','Austin'), ('TX','Houston'), ('NY','Albany'), 
             ('NY', 'Syracuse'), ('NY', 'Buffalo'),
             ('NY', 'Rochester'), ('TX', 'Dallas'), ('CA','Sacramento'), 
             ('CA', 'Palo Alto'), ('GA', 'Atlanta')]
cities_by_state = defaultdict(list)

for state, city in city_list:
    cities_by_state[state].append(city)
    
for state, cities in cities_by_state.items():
    print (state, ', '.join(cities))

'''
data = pos_triples = np.loadtxt(fname='fb_15k_train.tsv', dtype=str, comments='@Comment@ Subject Predicate Object')
data = pd.DataFrame(data, columns = ["subject","predicate","object"])

all_predicates = []
for i in range(0,len(data['predicate'])-1):
    p = data['predicate'][i]
    p = textwrap.dedent(p)
    splitted = re.split('/',p)
    results = list(filter(None, splitted))
    #print(type(results))
    #print(results)
    #for j in range(0,len(results)-1):
        #all_predicates.append(results[j])
    all_predicates.append(results)
#all_predicates = set(all_predicates)
#print(type(all_predicates))
#all_predicates = np.array(all_predicates)


model = Word2Vec(all_predicates)
X = model[model.wv.vocab]
print (list(model.wv.vocab))

from nltk.cluster import KMeansClusterer
from sklearn import cluster
from sklearn import metrics
import nltk

NUM_CLUSTERS = 5
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print(assigned_clusters)

words = list(model.wv.vocab)
for i, word in enumerate(words):
    print(word + ":" + str(assigned_clusters[i]))

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster id labels for inputted data")
print(labels)
print("Centroids data")
print(centroids)

print(
    "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print(kmeans.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

print("Silhouette_score: ")
print(silhouette_score)


#print (model.similarity('election', 'sports'))
#print (model.similarity('post', 'book'))

lookup = defaultdict(list)

#iterate through the training set and append the head and tail
for h,r,t in data.itertuples(index=False):
    print(r)
    lookup[(h,t)].append(r)
'''
match = []
counter = 0
for (h,t),r in lookup.items():
    print('running query for ', (h,t), ' to check whether if there is any mirror image for the same relation exist or not')
    counter = 0
    a = str((h,t))+ str(r)
    for (h,t),r in lookup.items():
        #print('looking for any further match')
        b = str((t,h))+  str(r)
        if (a == b):
            counter+=1
    if (counter>1):
        print((h,t),' matches with ',(t,h), 'in respect to relation')
counter = 0
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
    #print(splitted)

lookup2 = defaultdict()
lookup2 = dict.fromkeys(lookup1)
for (h,t),r in lookup1.items():
    if (len(r)>1):
        newlist = []
        for i in range(len(r)):
            for j in r[i]:
                if j not in newlist:
                    newlist.append(j)
    else:
        newlist = []
        for i in r[0]:
            if i not in newlist:
                newlist.append(i)
    lookup1[(h,t)].append(newlist)

for (h,t), r in lookup1.items():
    del (lookup1[(h,t)][0:-1])

from collections import OrderedDict
#sorted_lookpup1 = OrderedDict(sorted(len(lookup1.values())))
#sorted_lookpup1.sort(key=lambda x: sorted_lookpup1[x["eyecolor"]])
#ordered_d = OrderedDict(sorted(lookup1.viewitems(), key=lambda x: len(x[1])))

match = 0
indicator = 0
indicator2 =0
counter = 0
'''
for (h,t) in lookup1.items():
    print('outer', indicator)
    indicator2 = 0
    #print('running query for ',(h,t), ' to check whether if other relationships of same entity there or not')
    counter = 0
    a = str((h,t))
    for (h,t),r in lookup1.items():
        print('inner',indicator2)
        print(a)
        #print('looking for any further match')
        b = str((h, t))
        print('matching ', a, 'with ', b)
        if (a == b):
            #check for matching relation
            if(r!=r):
                print('found overlapping relations')
                counter+=1
        indicator2+=1
    indicator += 1
    if (counter>0):
        match+=1
        print('there are overlapping relations for ', (h,t))
if (match>0):
    print('there are overlapping relations in the dataset')
for (h,t),r in lookup1.items():
    print(r)
'''
for (h,t),r in lookup1.items():
    a = str((h,t))
    print(a)
    for (h,t), r in lookup1.items():
        b = str((h,t))
        print(b)
        print('matching ', a, 'with ', b)
        if (a == b):
            if (r!=r):
                print('found overlapping relations')
                counter+=1
    if(counter>0):
        print('there are overlapping relations')
if match>0:
    print('there are overlapping relations')