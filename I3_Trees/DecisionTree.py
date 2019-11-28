#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from random import sample

class DecisionTree:

    # Constructor
    def __init__(self, max_depth):
        self.max_depth = max_depth

    # Public method
    # Method for making decision tree
    def make_tree(self, df, depth=0, weight=None, weight_flag=False):

        if depth == 0:
            self.features = list(df.drop("Class", axis=1).columns)

        if self.__check_pure(df) == True or depth == self.max_depth:
            return self.__classify_leaf(df)
        else:
            split_on = self.__split_node(df, weight_flag)

            tree = {str(split_on): []}

            df_0 = df[df.eval(split_on) == 0]
            df_1 = df[df.eval(split_on) == 1]

            depth += 1

            ans_0 = self.make_tree(df_0, depth)
            ans_1 = self.make_tree(df_1, depth)

            tree[str(split_on)].append(ans_0)
            tree[str(split_on)].append(ans_1)

            return tree

    # Method for making decision tree for random forest
    def make_tree_rf(self, df, m_features, features_remain=None, depth=0):

        if depth == 0:
            features_remain = list(df.drop("Class", axis=1).columns)

        if self.__check_pure(df) == True or depth == self.max_depth:
            return self.__classify_leaf(df)
        else:

            # Here pick candidates, split based on best feature, and remove the best feature from all of features
            self.features = sample(features_remain, m_features)
            split_on = self.__split_node(df, weight_flag=False)
            features_remain.remove(str(split_on))

            tree = {str(split_on): []}

            df_0 = df[df.eval(split_on) == 0]
            df_1 = df[df.eval(split_on) == 1]

            depth += 1

            # Re-sampling for the next right and left node
            ans_0 = self.make_tree_rf(df_0, m_features, features_remain=features_remain, depth=depth)
            ans_1 = self.make_tree_rf(df_1, m_features, features_remain=features_remain, depth=depth)

            tree[str(split_on)].append(ans_0)
            tree[str(split_on)].append(ans_1)

            return tree

    # Method for making decision tree for AdaBoost
    def make_tree_adaboost(self, df, depth=0, weight_flag=True):

        if depth == 0:
            self.features = list(df.drop(["weight", "Class"], axis=1).columns)

        if self.__check_pure(df) == True or depth == self.max_depth:
            return self.__classify_leaf(df, weight_flag)
        else:

            split_on = self.__split_node(df, weight_flag)

            tree = {str(split_on): []}

            df_0 = df[df.eval(split_on) == 0]
            df_1 = df[df.eval(split_on) == 1]

            depth += 1

            ans_0 = self.make_tree_adaboost(df_0, depth)
            ans_1 = self.make_tree_adaboost(df_1, depth)

            tree[str(split_on)].append(ans_0)
            tree[str(split_on)].append(ans_1)

            return tree

    # Method for getting accuracy of the prediction
    def accuracy(self, df, model_tree):

        y_labels = df["Class"].to_list()
        predictions = []
        correct = 0

        for i, example in enumerate(df.iterrows()):
            df_example = df[df.index == i].drop("Class", axis=1)
            y_pred = self.predict(df_example, model_tree)
            predictions.append(y_pred)

        for i in range(len(predictions)):
            if predictions[i] == y_labels[i]:
                correct += 1

        accuracy = 100 * correct / len(predictions)

        return accuracy

    # Method for getting prediction result
    def predict(self, x_test, y_predict):

        if type(y_predict) == int:
            pass
        elif type(y_predict) == dict:
            feature = list(y_predict.keys())[0]
            feature_example = x_test.eval(feature).values[0]
            y_predict = y_predict[str(feature)][feature_example]
            y_predict = self.predict(x_test, y_predict)

        else:
            print("error: not int or dict")

        return y_predict

    # Private method
    # Method for checking impurity
    def __check_pure(self, df):
        labels = np.unique(df.Class.values)
        if len(labels) == 1:
            return True
        else:
            return False

    def __classify_leaf(self, df, weight_flag=False):
        if weight_flag == False:
            n_1 = sum(df["Class"])
            n_0 = len(df) - n_1
            if n_1 > n_0:
                return 1
            else:
                return 0
        elif weight_flag == True:

            weight = df["weight"].values
            Class = df["Class"].values

            sum_w0 = 0
            sum_w1 = 0

            for i in range(len(df)):

                if Class[i] == 0:
                    sum_w0 += weight[i]
                elif Class[i] == 1:
                    sum_w1 += weight[i]
            if sum_w1 > sum_w0:
                return 1
            else:
                return 0

    def __split_node(self, df, weight_flag):

        benefits = [self.__get_benefit(df, feature, weight_flag) for feature in self.features]

        try:
            split_on = self.features[np.nanargmax(benefits)]
        except:
            split_on = np.NaN

        return split_on

    def __get_benefit(self, df, feature, weight_flag):

        if len(df) == 0:
            return 0
        else:

            U_A = self.__get_uncertainty(df, weight_flag)
            p_left = self.__get_prob(df, feature, 0)
            p_right = self.__get_prob(df, feature, 1)

            U_AL = self.__get_uncertainty(df[df.eval(feature) == 0], weight_flag)
            U_AR = self.__get_uncertainty(df[df.eval(feature) == 1], weight_flag)

            benefit = U_A - (p_left * U_AL) - (p_right * U_AR)

            return benefit

    def __get_uncertainty(self, df, weight_flag=False):

        if weight_flag == False:
            n_1 = sum(df["Class"])
            n_0 = len(df) - n_1
            n = len(df)

            try:
                U = 1 - (float(n_1 / n)) ** 2 - (float(n_0 / n)) ** 2
            except:
                U = 0 # Use zero value instead of nan

            return U

        else:

            weight = df["weight"].values
            df = df.drop("weight", axis=1)

            # Obtain correct label
            correct_label = df["Class"].to_list()

            # Initialization
            sum_w_1 = 0
            sum_w_0 = 0

            # Loop for weight summation
            for i in range(len(correct_label)):
                if correct_label[i] == 1:
                    sum_w_1 += weight[i]
                elif correct_label[i] == 0:
                    sum_w_0 += weight[i]

            sum_w = sum_w_1 + sum_w_0

            try:
                U = 1 - (float(sum_w_1 / sum_w)) ** 2 - (float(sum_w_0 / sum_w)) ** 2
            except:
                U = 0 # Use zero value instead of nan

            return U

    def __get_prob(self, df, feature, feature_result):

        n_1 = sum(df["Class"])
        n_0 = len(df) - n_1

        branch = df[df.eval(feature) == feature_result]
        branch_n_1 = sum(branch["Class"])
        branch_n_0 = len(branch) - branch_n_1

        try:
            prob = (branch_n_1 + branch_n_0) / (n_1 + n_0)
        except:
            prob = np.NaN

        return prob