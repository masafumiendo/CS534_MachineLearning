#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from random import sample

import PreProcess
import DecisionTree

class RandomForest:

    def __init__(self, n_trees, class_learner, m_features, max_depth=2):

        self.n_trees = n_trees
        self.class_learner = class_learner
        self.learners = [self.class_learner for _ in range(self.n_trees)]
        self.m_features = m_features
        self.max_depth = max_depth
        self.trees = []

    # Public method
    # Method for making multiple decision trees
    def make_trees(self, df):

        for t in range(self.n_trees):
            df_bootstrap = self.__bootstrap(df)
            tree = self.learners[t].make_tree_rf(df_bootstrap, self.m_features)

            self.trees.append(tree)

    # Method for getting accuracy by voting
    def accuracy(self, df):

        y_labels = df["Class"].to_list()
        predictions = []

        for i, example in enumerate(df.iterrows()):
            df_ex = df[df.index == i].drop("Class", axis=1)
            votes = []

            for j in range(self.n_trees):

                vote = self.learners[j].predict(df_ex, self.trees[j])
                votes.append(vote)

            y_pred = max(set(votes), key=votes.count)
            predictions.append(y_pred)

        correct = 0
        for k in range(len(predictions)):
            if predictions[k] == y_labels[k]:
                correct += 1

        accuracy = 100 * correct / len(predictions)

        return accuracy

    def plot(self, train, valid, variable, parameter):

        plt.plot(variable, train, label="training")
        plt.plot(variable, valid, label="validation")
        plt.xlabel("number of " + parameter)
        plt.ylabel("accuracy [%]")
        plt.ylim((0, 100))
        plt.title("training and validation accuracy")
        plt.legend()

        plt.savefig('fig/part2/train_valid_acc_{0}.png'.format(parameter))

    def __bootstrap(self, df):
        return df.sample(frac=1, replace=True, random_state=42)


def main():

    preprocess = PreProcess.PreProcess()

    df_train, df_valid, df_test = preprocess.get_data()

    train_accs = []
    valid_accs = []
    n_trees = [1, 2, 5, 10, 25] # 
    m_features = [1, 2, 5, 10, 25, 50]

    # Loop for number of trees
    for n in n_trees:
        print(n)
        RF = RandomForest(n_trees=n, class_learner=DecisionTree.DecisionTree(2), m_features=5)
        RF.make_trees(df_train)

        train_acc = RF.accuracy(df_train)
        valid_acc = RF.accuracy(df_valid)

        print("training accuracy: {0}, validation accuracy: {1} with {2} decision trees".format(train_acc, valid_acc, n))

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

    RF.plot(train_accs, valid_accs, n_trees, "trees")

    train_accs = []
    valid_accs = []

    # Loop for number of features
    for m in m_features:
        RF = RandomForest(n_trees=15, class_learner=DecisionTree.DecisionTree(2), m_features=m)
        RF.make_trees(df_train)

        train_acc = RF.accuracy(df_train)
        valid_acc = RF.accuracy(df_valid)

        print("training accuracy: {0}, validation accuracy: {1} with {2} features".format(train_acc, valid_acc, m))

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

    RF.plot(train_accs, valid_accs, m_features, "features")


if __name__ == '__main__':
    main()