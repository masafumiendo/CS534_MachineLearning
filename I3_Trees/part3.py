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
        # y = self.__label_converter(y)

        N = len(x)
        weight = np.ones(N) / N # Initialize the weight
        df['weight'] = weight

        for i in range(self.num_learners):

            tree = self.learners[i].make_tree_adaboost(df)
            print(weight)
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
            print("epsilon", epsilon)
            self.alpha[i] = 1/2 * np.log((1.0 - epsilon)/epsilon)

            # Update the weight
            weight = df["weight"].values
            for j in range(len(weight)):
                if mistakes[j] == True:
                    # print(j)
                    weight[j] = weight[j] * np.exp(self.alpha[i])
                elif mistakes[j] == False:
                    weight[j] = weight[j] * np.exp(-self.alpha[i])
            print("n mistakes", sum(mistakes))

            weight = weight / np.sum(weight)
            df['weight'] = weight
            # print(df.columns)
            #print(sum(weight))
            print("alpha ", self.alpha)
           # print(np.unique(weight))

    # Method for prediction
    def predict(self, df):

        base_pred = np.zeros((self.num_learners, len(df)))
        for i in range(self.num_learners):
            for j, example in enumerate(df.iterrows()):
                df_ex = df[df.index == j].drop("Class", axis=1)
                prediction = self.learners[i].predict(df_ex, self.trees[i])
                if prediction == 0:
                    prediction = -1
                base_pred[i][j] = prediction
            #print(base_pred[i])
            print(np.sign(base_pred.T @ self.alpha))
        self.alpha = self.alpha / np.sum(self.alpha)
        print(np.sign(base_pred.T @ self.alpha))
        return np.sign(base_pred.T @ self.alpha)

    # Method for computing accuracy
    def accuracy(self, prediction, df):

        y = df["Class"].to_list()
        y = self.__label_converter(y)
       # prediction = self.__label_converter(prediction)
        # print(self.alpha)
        cnt = 0

        for i in range(len(y)):
            if y[i] == prediction[i]:
                cnt += 1
            else:
                pass

        print("count", cnt)
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

if __name__ == '__main__':

    preprocess = PreProcess.PreProcess()

    df_train, df_valid, df_test = preprocess.get_data()

    # DT = DecisionTree.DecisionTree(2) # with max depth is one
    #
    # adaboost = AdaBoost(DT, 5) # The number of weak leaners
    #
    # adaboost.train(df_train)
    #
    # train_pred = adaboost.predict(df_train)
    # print(train_pred)
    # train_acc = adaboost.accuracy(train_pred, df_train)
    #
    # valid_predict = adaboost.predict(df_valid)
    # accuracy = adaboost.accuracy(valid_predict, df_valid)
    #
    # print("valid acc", accuracy)
    # print("train acc:", train_acc)

    n_learners = [1, 2, 5, 10, 15]
    train_accs = []
    valid_accs = []
    for n_learner in n_learners:
        DT = DecisionTree.DecisionTree(1)  # with max depth is one
        adaboost = AdaBoost(DT, n_learner)  # The number of weak leaners
        adaboost.train(df_train)

        train_pred = adaboost.predict(df_train)
        valid_pred = adaboost.predict(df_valid)

        train_acc = adaboost.accuracy(train_pred, df_train)
        valid_acc = adaboost.accuracy(valid_pred, df_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

    learners_plot(100*np.array(train_accs), 100*np.array(valid_accs), n_learners)