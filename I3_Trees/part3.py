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

    # Public method
    # Method for training weak learners
    def train(self, df):

        x, y = df # Need to split

        N = len(x)
        weight = np.ones(N) / N

        for i in range(self.num_learners):
            tree = self.learners[i].make_tree(df)
            mistakes = (self.learners[i].predict(x, tree) != y)

            # Compute the epsilon and alpha
            epsilon = np.sum(weight * mistakes)
            self.alpha[i] = np.log(1.0/epsilon - 1)

            # Update the weight
            weight = weight * np.exp(self.alpha[i] * mistakes)
            weight = weight / sum(weight)

    # Method for prediction
    def predict(self, df):

        x, y = df # Need to split

        base_pred = np.zeros((self.num_learners, len(x)))
        for i in range(self.num_learners):
            base_pred[i] = self.learners[i].predict()

        return np.sign(base_pred.T @ self.alpha)

def main():

    preprocess = PreProcess.PreProcess()

    df_train, df_valid, df_test = preprocess.get_data()

    DT = DecisionTree.DecisionTree(1)

    adaboost = AdaBoost(DT, 5)

if __name__ == '__main__':
    main()