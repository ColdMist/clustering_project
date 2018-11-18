#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:55:46 2018

@author: turzo
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display

from utilities import util

def read_and_replace(fpath, def_df):
    df = pd.read_table(fpath, names=['head', 'rel', 'tail'])
    df['head'] = def_df.loc[df['head']]['word'].values
    df['tail'] = def_df.loc[df['tail']]['word'].values
    return df

#Load the data and create a pandas dataframe object
data_dir = 'data/wordnet-mlj12' # change to where you extracted the data
definitions = pd.read_table(os.path.join(data_dir, 'wordnet-mlj12-definitions.txt'), 
                            index_col=0, names=['word', 'definition'])
train = read_and_replace(os.path.join(data_dir, 'wordnet-mlj12-train.txt'), definitions)
val = read_and_replace(os.path.join(data_dir, 'wordnet-mlj12-valid.txt'), definitions)
test = read_and_replace(os.path.join(data_dir, 'wordnet-mlj12-test.txt'), definitions)

#Look at the data to get an overview
print('Train shape:', train.shape)
print('Validation shape:', val.shape)
print('Test shape:', test.shape)
all_train_entities = set(train['head']).union(train['tail'])
print('Training entity count: {}'.format(len(all_train_entities)))
print('Training relationship type count: {}'.format(len(set(train['rel']))))
print('Example training triples:')
display(train.sample(5))

'''
Since most of the relation in the triples of the validation set and test sets are mirror images of the training set, that is
why we are going to remove that too.
'''
#we will find the statndat nonlinearity based on the function
from collections import defaultdict

#Now create a mask varialbe for removing purpose
mask = np.zeros(len(train)).astype(bool)
lookup = defaultdict(list)

#iterate through the training set and append the head and tail 
for idx,h,r,t in train.itertuples():
    lookup[(h,t)].append(idx) # try to use lookup default dictionary to append the indexes to the head, tail as index
#Combine the training validation and test set using concat fucntion of dataframe
#train_val_test_combined = pd.concat((train,val,test))
#Build the mask with the indicies of the validation and test set so that we can remove them from training set
for h,r,t in pd.concat((val,test)).itertuples(index=False):
    mask[lookup[(h,t)]] = True
    mask[lookup[(t,h)]] = True
#Remove the masked rows from the training set
train = train.loc[~mask]
heads,tails = set(train['head']), set(train['tail'])
val = val.loc[val['head'].isin(heads) & val['tail'].isin(tails)]
test = test.loc[test['head'].isin(heads) & test['tail'].isin(tails)]
'''
We will now creates some false statemens to make it as a classification problem [Socher13]. For each true statement corrupt it by either 
either replacing the head or tail with a random checking.
'''
#Here we will use our own create corrupt sample creation

rng = np.random.RandomState(42)
combined_df = pd.concat((train, val, test))
val = util.create_tf_pairs(val, combined_df, rng)
test = util.create_tf_pairs(test, combined_df, rng)
print('Validation shape:', val.shape)
print('Test shape:', test.shape)
#For testing purpose
#rng = np.random.RandomState(42)
#combined_df = pd.concat((train, val, test))

#val_res = util.create_tf_pairs(val, combined_df, rng)
#test_res = util.create_tf_pairs(test, combined_df, rng)

#Lets check what kind of prediction task we are up against, lets examine the training and test data for involving the entity 'brain_cell'
example_entity = '__brain_cell_NN_1'
example_train_rows = (train['head'] == example_entity) | (train['tail'] == example_entity)
print('Train: ')
display(train.loc[example_train_rows])
example_test_rows = (test['head'] == example_entity) | (test['tail'] == example_entity)
print('Test: ')
display(test.loc[example_test_rows])


# we will do matrix factorization right away
has_part = val.loc[val['rel']] == 'has_part'
has_part = val.loc[val['head']] == '__retina_NN_1'
has_part_triples = val.loc[val['rel'] == '_has_part']
query_entities = ['__noaa_NN_1', '__vascular_plant_NN_1', '__retina_NN_1']
has_part_example = has_part_triples.loc[has_part_triples['head'].isin(query_entities)]
matrix_view = pd.pivot_table(has_part_example, 'truth_flag', 'head', 'tail', 
                             fill_value=False).astype(int)
display(matrix_view)

