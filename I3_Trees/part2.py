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
from random import sample
from part1 import DecisionTree as DecisionTree


class RandomForest:

    def __init__(self, n_trees=1, m_features=100, max_depth=2):

        self.n_trees = n_trees
        self.m_features = m_features
        self.max_depth = max_depth

    def make_n_trees(self, df_train):
        trees = []

        for t in range(self.n_trees):
            df = self.bootstrap(df_train)
            df = self.get_m_features(df)
            DT = DecisionTree(max_depth=self.max_depth)
            tree = DT.make_decisiontree(df)
            trees.append(tree)

        return trees

    def bootstrap(self, df):
        # sample len(df) with replacement
        new_df = df.sample(frac=1, replace=True, random_state=42)
        return new_df

    def get_m_features(self, df):
        # sample m features for tree
        features = list(df.drop("Class", axis=1).columns)
        new_features = sample(features, self.m_features)
        new_features.append("Class")
        new_df = df[new_features]
        return new_df

    def accuracy(self, df, model_trees):

        y_labels = df["Class"].to_list()
        #        print(y_labels)
        predictions = []
        for i, example in enumerate(df.iterrows()):
            df_ex = df[df.index == i].drop("Class", axis=1)  # single example as df
            votes = []
            for tree in model_trees:
                DT = DecisionTree(max_depth=self.max_depth)  # because .predict needs self
                vote = DT.predict(df_ex, tree)  # prediction of example using one tree
                votes.append(vote)
            y_pred = max(set(votes), key=votes.count)
            predictions.append(y_pred)

        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_labels[i]:
                #                print(predictions[i], y_labels[i])
                correct += 1
        accuracy = 100 * correct / len(predictions)  # accuracy as percent correct
        return accuracy

        return accuracy

    def plot_train_valid_accuracy(self, train, valid, n_trees_set):
        plt.plot(n_trees_set, train, label="training")
        plt.plot(n_trees_set, valid, label="validation")

        plt.ylim((0, 100))

        plt.xlabel("Number of trees")
        plt.ylabel("Accuracy [%]")

        plt.title("Train and validation accuracy")
        plt.legend()
        #        plt.show()
        plt.savefig('fig/part2/2b_vary_trees_train_validation_accuracy.png')


#        plt.close(fig)


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
    df_valid = df_valid.drop(cols_to_drop, axis=1)
    df_test = df_test.drop(cols_to_drop, axis=1)

    #    rf = RandomForest(n_trees = 5, m_features = 100, max_depth = 2)
    #    new_df = rf.bootstrap(df_train)
    #    newer_df = rf.get_m_features(new_df)
    #
    #    model = rf.make_n_trees(df_train)
    #
    #    valid_acc = rf.accuracy(df_valid, model)
    #    train_acc = rf.accuracy(df_train, model)
    #    print(train_acc, valid_acc)
    #

    # part 2b
    # plot train and validation accuracy for n_trees_set
    train_acc = []
    valid_acc = []
    n_trees_set = [1, 2, 5, 10, 25]
    for n in n_trees_set:
        rf = RandomForest(n_trees=n, m_features=5, max_depth=2)
        model_tree = rf.make_n_trees(df_train)

        train_accuracy = rf.accuracy(df_train, model_tree)
        valid_accuracy = rf.accuracy(df_valid, model_tree)
        print(train_accuracy, valid_accuracy)

        train_acc.append(train_accuracy)
        valid_acc.append(valid_accuracy)

    rf.plot_train_valid_accuracy(train_acc, valid_acc, n_trees_set)