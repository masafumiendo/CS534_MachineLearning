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

class DecisionTree:

    def __init__(self, df_train, max_depth):
    
        self.max_depth = max_depth
        self.autoencoded_features = list(df_train.columns)[:-1]
        self.features = list(np.unique([str(feature).split("_")[0] for feature in self.autoencoded_features]))
    
    def make_decisiontree(self, df, depth=0):
        
        if self.check_pure(df) == True or depth == self.max_depth:
#            print("leaf")
            return self.classify_leaf(df) #returns 0 or 1 for classification
            
        else:
            split_on = self.split_node(df)
            tree = {str(split_on): []}
            
            df_0 = df[df.eval(split_on) == 0]
            df_1 = df[df.eval(split_on) == 1]
            
            depth += 1
            print(depth)
            
            ans_0 = self.make_decisiontree(df_0, depth)
            ans_1 = self.make_decisiontree(df_1, depth)
            
            tree[str(split_on)].append(ans_0)
            tree[str(split_on)].append(ans_1)
            
            return tree   
    
    def check_pure(self, df):
        labels = np.unique(df.Class.values)
        if len(labels) == 1:
            return True
        else:
            return False
    
    def classify_leaf(self, df):
        n_1 = sum(df["Class"])
        n_0 = len(df) - n_1
        if n_1 > n_0:
            return 1 # return poisonous
        else:
            return 0 # return edible
        
    def split_node(self, df):
        try:
            benefits = [self.get_benefit(df, feature) for feature in self.autoencoded_features]
            split_on = self.autoencoded_features[np.argmax(benefits)] # split on feature with max benefit value
        except:
            split_on = 0
        return split_on
    
    def get_benefit(self, df, feature):
#        print(feature)
        if len(df) == 0:
            return 0
        else:
            U_A = self.get_uncertainty(df)
            p_left = self.get_probability(df, feature, 0)
            p_right = self.get_probability(df, feature, 1)
            try: 
                U_AL = self.get_uncertainty(df[df.eval(feature) == 0])
                U_AR = self.get_uncertainty(df[df.eval(feature) == 1])
                benefit = U_A - (p_left*U_AL) - (p_right * U_AR)
            except: # if all of branch has 0 or 1 for feature, so no benefit splitting
                benefit = np.NaN # should not happen bc of check_pure function
            return benefit
        
    def get_uncertainty(self, df):
        
        n_1 = sum(df["Class"])
        n_0 = len(df) - n_1
        n_total = len(df)

        try:
            U = 1 - (float(n_1/n_total))**2 - (float(n_0/n_total))**2
        except:
            U = np.NaN
        return U
    
    def get_probability(self, df, feature, feature_result):
        
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
    
    def valid_accuracy(self, df_valid, model_tree):
        y_labels = df_valid["Class"].to_list()
        predictions = []
        for i, example in enumerate(df_valid.iterrows()):
            df_ex = df_valid[df_valid.index == i].drop("Class", axis = 1)
            y_pred = self.predict(df_ex, model_tree)
            predictions.append(y_pred)
        correct = sum([(predictions[i] == y_labels[i]) for i in predictions])
        accuracy = correct / len(predictions)
        return accuracy
    
    def predict(self, x_test, y_predict):
        
        if type(y_predict) == int:
            print("y_predict:", y_predict)
          
        elif type(y_predict) == dict:
            feature = list(y_predict.keys())[0]
            example_feature = x_test.eval(feature).values[0]
            y_predict = y_predict[str(feature)][example_feature]
            y_predict = self.predict(x_test, y_predict)
            
        else:
            print("not int or dict")
        
        return y_predict

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

    # make decision tree model
    max_depth = 5
    DT = DecisionTree(df_train, max_depth)
    model_tree = DT.make_decisiontree(df_train)

    # predict example
    x_test = df_test[df_test.index == 78]
    y_pred = DT.predict(x_test, model_tree)
    
    # validation accuracy
    accuracy = DT.valid_accuracy(df_valid, model_tree)
    