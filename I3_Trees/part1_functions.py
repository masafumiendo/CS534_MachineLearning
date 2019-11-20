#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:55:31 2019

@author: mayermo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

def make_decisiontree(df, features, max_depth, depth=0):
    
    if check_pure(df) == True or depth == max_depth:
#        print(check_pure(df), depth, max_depth)
        return classify_leaf(df) #returns 0 or 1 for classification
        
    else:
        split_on = split_node(df, features)
        print(split_on)
        tree = {str(split_on): []}
        
        df_0 = df[df.eval(split_on) == 0]
        df_1 = df[df.eval(split_on) == 1]
#        df_0 = df.drop(df.index[df[str(split_on)] == 1], inplace = True)
#        df_1 = df.drop(df.index[df[str(split_on)] == 0], inplace = True)
#        print(df_1)
        print(len(df_0), len(df_1))
        
        depth += 1
        print(depth)
        
        ans_0 = make_decisiontree(df_0, features, max_depth, depth)
        ans_1 = make_decisiontree(df_1, features, max_depth, depth)
        
        tree[str(split_on)].append(ans_0)
        tree[str(split_on)].append(ans_1)
        
        return tree   

def check_pure(df):
    labels = np.unique(df.Class.values)
#    print(df)
#    print(labels)
    if len(labels) == 1:
        return True
    else:
        return False

def classify_leaf(df):
    n_1 = sum(df["Class"])
    n_0 = len(df) - n_1
    if n_1 > n_0:
        return 1 # return poisonous
    else:
        return 0 # return edible
    
def split_node(df, features):
#    try:
    benefits = [get_benefit(df, feature) for feature in features]
    print(benefits)
    print(max(benefits))
    split_on = features[np.argmax(benefits)] # split on feature with max benefit value
#    except:
#        split_on = 0
    return split_on

def get_benefit(df, feature):
#        print(feature)

    U_A = get_uncertainty(df)
    p_left = get_probability(df, feature, 0)
    p_right = get_probability(df, feature, 1)
 
    U_AL = get_uncertainty(df[df.eval(feature) == 0])
    U_AR = get_uncertainty(df[df.eval(feature) == 1])
    
    try:
        benefit = U_A - (p_left*U_AL) - (p_right * U_AR)
    except: # if all of branch has 0 or 1 for feature, so no benefit splitting
        benefit = np.NaN # should not happen bc of check_pure function
    
    return benefit
    
def get_uncertainty(df):
    
    n_1 = sum(df["Class"])
    n_0 = len(df) - n_1
    n_total = len(df)
#    print(n_total)
    try:
        U = 1 - (float(n_1/n_total))**2 - (float(n_0/n_total))**2
    except:
        U = np.NaN
    return U

def get_probability(df, feature, feature_result):
    
    n_1 = sum(df["Class"])
    n_0 = len(df) - n_1
    
    #branch is df of feature = 0 or 1 (result)
    branch = df[df.eval(feature) != feature_result]
    branch_n_1 = sum(branch["Class"])
    branch_n_0 = len(branch) - branch_n_1
    
    try:
        probability = (branch_n_1 + branch_n_0)/(n_1 + n_0)
    except:
        probability = np.NaN
    return probability
    
        
if __name__ == "__main__":
    
    df_train = pd.read_csv('pa3_train.csv')
    df_valid = pd.read_csv('pa3_val.csv')
    df_test = pd.read_csv('pa3_test.csv')
    
    # change column names to remove '-' from col names because of pandas issue, also class-> Class bc of class object, replace '?'
    df_train.columns = [col.replace("-", "").replace("class", "Class").replace("?", "unk") for col in df_train.columns]
    df_valid.columns = [col.replace("-", "").replace("class", "Class").replace("?", "unk") for col in df_valid.columns]
    df_test.columns = [col.replace("-", "").replace("?", "unk") for col in df_test.columns]
    
    features = list(df_train.columns)[:-1]

    max_depth = 3
    model_tree = make_decisiontree(df_train, features, max_depth)

