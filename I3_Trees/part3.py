#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import PreProcess
import DecisionTree

class AdaBoost:

    def __init__(self, class_learner, num_learners):
        self.class_learner = class_learner
        self.num_learners = num_learners
        self.learners = [self.class_learner for _ in range(self.num_learners)]
        self.alpha = np.zeros(self.num_learners)
        self.trees = []

    # Public method
    # Method for training weak learners
    def train(self, df):

        x = df.drop("Class", axis=1)
        y = df["Class"].to_list()
        y = self.__label_converter(y)

        N = len(x)
        weight = np.ones(N) / N # Initialize the weight

        for i in range(self.num_learners):
            tree = self.learners[i].make_tree(df)
            mistakes = (self.learners[i].predict(x, tree) != y)

            self.trees.append(tree)

            # Compute the epsilon and alpha
            epsilon = np.sum(weight * mistakes)
            self.alpha[i] = np.log(1.0/epsilon - 1)

            # Update the weight
            weight = weight * np.exp(self.alpha[i] * mistakes)
            weight = weight / np.sum(weight)

    # Method for prediction
    def predict(self, df):

        x = df.drop("Class", axis=1)

        base_pred = np.zeros((self.num_learners, len(x)))
        for i in range(self.num_learners):
            base_pred[i] = self.learners[i].predict(x, self.trees[i])

        return np.sign(base_pred.T @ self.alpha)

    def __label_converter(self, y):

        num_row = len(y)
        y_convert = np.zeros(len(y))

        for i in range(num_row):
            if y[i] == 1:
                y_convert[i] = 1
            elif y[i] == 0:
                y_convert[i] = -1

        return y_convert

def main():

    preprocess = PreProcess.PreProcess()

    df_train, df_valid, df_test = preprocess.get_data()

    DT = DecisionTree.DecisionTree(1) # with max depth is one

    adaboost = AdaBoost(DT, 10) # The number of weak leaners

    adaboost.train(df_train)

if __name__ == '__main__':
    main()