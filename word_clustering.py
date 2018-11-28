#Word clustering based on doc2vec and Kmeans
############################################


import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
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

#######################################################################
def splitter(words):
    splitted_word_list = []
    for word in words:
        splitted_word = re.split('_',word)
        splitted_word_list.append(splitted_word)
    return splitted_word_list

relation_list = splitter(relation_list)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(relation_list)

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")
########################################################################
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
'''
data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]
'''
flat_relation_list = [i[0] for i in relation_list]
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(flat_relation_list)]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("Sports".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])
X =model.docvecs
textVect = model.docvecs.doctag_syn0

## K-means ##
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(textVect)
clusters = km.labels_.tolist()

## Print Sentence Clusters ##
cluster_info = {'sentence': flat_relation_list, 'cluster' : clusters}
sentenceDF = pd.DataFrame(cluster_info, index=[clusters], columns = ['sentence','cluster'])
for num in range(num_clusters):
    print()
    print("Sentence cluster %d: " %int(num+1), end='')
    print()
    for sentence in sentenceDF.ix[num]['sentence'].values.tolist():
        print(' %s ' %sentence, end='')
        print()
        print()

cluster_dict = dict()
for cluster, relation in zip(clusters,flat_relation_list):
    cluster_dict[relation] = cluster
    #print(cluster, relation)

######################################################
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