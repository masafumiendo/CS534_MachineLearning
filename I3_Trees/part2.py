#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:06:15 2019

@author: morgan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from part1 import DecisionTree as DecisionTree

class RandomForest:
    
    def __init__(self, n_trees, n_features, max_depth):
    
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        
    def bootstrap(self, df):
        # sample len(df) with replacement
        sample_indices = []
        for a in range(len(df)):
#        if len(sample_indices) < len(df):
            index = np.random.randint(len(df))
            sample_indices.append(index)
        new_df = df[df.index in sample_indices]
        
        return new_df
        


if __name__ == "__main__":
    
    df_train = pd.read_csv('pa3_train.csv')
    df_valid = pd.read_csv('pa3_val.csv')
    df_test = pd.read_csv('pa3_test.csv')
    
    # change column names to remove '-' from col names because of pandas issue, also class-> Class bc of class object, replace '?'
    df_train.columns = [col.replace("-", "").replace("class", "Class").replace("?", "unk") for col in df_train.columns]
    df_valid.columns = [col.replace("-", "").replace("class", "Class").replace("?", "unk") for col in df_valid.columns]
    df_test.columns = [col.replace("-", "").replace("?", "unk") for col in df_test.columns]
    
    # remove columns where value is same for all molecules
    nunique = df_train.apply(pd.Series.nunique)
    cols_to_drop = list(nunique[nunique == 1].index)
    df_train = df_train.drop(cols_to_drop, axis=1)
    df_valid = df_valid.drop(cols_to_drop, axis = 1)
    df_test = df_test.drop(cols_to_drop, axis = 1)

    rf = RandomForest(1, 2, 3)