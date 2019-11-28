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
        self.trees = []

    # Public method
    # Method for training weak learners
    def train(self, df):

        x = df.drop("Class", axis=1)
        y = df["Class"].to_list()

        N = len(x)
        weight = np.ones(N) / N # Initialize the weight
        df['weight'] = weight
        alpha = np.zeros(self.num_learners)

        for i in range(self.num_learners):

            tree = self.learners[i].make_tree_adaboost(df)
            mistakes = np.zeros(len(df))

            for j, example in enumerate(df.iterrows()):
                df_ex = df[df.index == j].drop(["Class", "weight"], axis=1)
                prediction = self.learners[i].predict(df_ex, tree)
                if prediction == y[j]:
                    mistakes[j] = False
                elif prediction != y[j]:
                    mistakes[j] = True

            self.trees.append(tree)

            # Compute the epsilon and alpha
            epsilon = np.sum(weight * mistakes)

            alpha[i] = 1/2 * np.log((1.0 - epsilon)/epsilon)

            # Update the weight
            weight = df["weight"].values
            for j in range(len(weight)):
                if mistakes[j] == True:
                    weight[j] = weight[j] * np.exp(alpha[i])
                elif mistakes[j] == False:
                    weight[j] = weight[j] * np.exp(-alpha[i])

            weight = weight / np.sum(weight)
            df['weight'] = weight

        return alpha

    # Method for prediction by a strong learner
    def predict(self, df, alpha, test=False):

        base_pred = np.zeros((self.num_learners, len(df)))

        for i in range(self.num_learners):
            for j, example in enumerate(df.iterrows()):
                if test==False:
                    df_ex = df[df.index == j].drop("Class", axis=1)
                else:
                    df_ex = df[df.index == j]
                prediction = self.learners[i].predict(df_ex, self.trees[i])
                if prediction == 0:
                    prediction = -1
                base_pred[i][j] = prediction

        alpha = alpha / np.sum(alpha)

        voted_pred = np.sign(base_pred.T @ alpha)

        return voted_pred

    # Method for computing accuracy
    def accuracy(self, prediction, df):

        y = df["Class"].to_list()
        y = self.__label_converter(y)

        cnt = 0

        for i in range(len(y)):
            if y[i] == prediction[i]:
                cnt += 1
            else:
                pass

        accuracy = cnt / len(y)

        return accuracy

    def __label_converter(self, y):

        num_row = len(y)
        y_convert = np.zeros(len(y))

        for i in range(num_row):
            if y[i] == 1:
                y_convert[i] = 1
            elif y[i] == 0:
                y_convert[i] = -1

        return y_convert

def learners_plot(train, valid, n_learners):
    
    fig = plt.figure()

    plt.plot(n_learners, train, label="training")
    plt.plot(n_learners, valid, label="validation")
    plt.xlabel("number of weak learners")
    plt.ylabel("accuracy [%]")
    plt.ylim((50, 100))
    plt.title("training and validation accuracy")
    plt.legend()

    plt.savefig("fig/part3/n_learners_train_valid_acc.png")
    
    plt.clf()

def main():

    preprocess = PreProcess.PreProcess()

    df_train, df_valid, df_test = preprocess.get_data()

    n_learners = [1, 2, 5, 10, 15]
    train_accs = []
    valid_accs = []

    for n_learner in n_learners:
        DT = DecisionTree.DecisionTree(1)  # with max depth is one
        adaboost = AdaBoost(DT, n_learner)  # The number of weak leaners
        alpha = adaboost.train(df_train)

        train_pred = adaboost.predict(df_train, alpha)
        valid_pred = adaboost.predict(df_valid, alpha)

        train_acc = adaboost.accuracy(train_pred, df_train)
        valid_acc = adaboost.accuracy(valid_pred, df_valid)

        print("training accuracy: {0}, validation accuracy: {1} for num. of leaners of {2}".format(train_acc, valid_acc, n_learner))
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

    learners_plot(100*np.array(train_accs), 100*np.array(valid_accs), n_learners)

    n_learner = 6

    DT = DecisionTree.DecisionTree(2)
    adaboost = AdaBoost(DT, n_learner)
    alpha = adaboost.train(df_train)

    train_pred = adaboost.predict(df_train, alpha)
    valid_pred = adaboost.predict(df_valid, alpha)

    train_acc = adaboost.accuracy(train_pred, df_train)
    valid_acc = adaboost.accuracy(valid_pred, df_valid)

    print("training accuracy: {0}, validation accuracy: {1} for num. of leaners of {2}".format(train_acc, valid_acc, n_learner))

    test_pred = adaboost.predict(df_test, alpha, True)
    test_pred = pd.DataFrame(test_pred)
    test_pred.to_csv('pa3_res.csv', header=False, index=False)

if __name__ == '__main__':

    main()
