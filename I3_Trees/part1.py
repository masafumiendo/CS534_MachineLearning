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

import PreProcess
import DecisionTree

def plot(train, valid):

    plt.plot(np.arange(1, 1 + len(train)), train, label="training")
    plt.plot(np.arange(1, 1 + len(valid)), valid, label="validation")
    plt.xlabel("maximum depth")
    plt.ylabel("accuracy [%]")
    plt.ylim((80, 100))
    plt.title("training and validation accuracy")
    plt.legend()

    plt.savefig("fig/part1/train_valid_acc.png")

def main():

    preprocess = PreProcess.PreProcess()

    df_train, df_valid, df_test = preprocess.get_data()

    train_accs = []
    valid_accs = []

    for i in range(1, 9):
        DT = DecisionTree.DecisionTree(i)
        tree = DT.make_tree(df_train)

        train_acc = DT.accuracy(df_train, tree)
        valid_acc = DT.accuracy(df_valid, tree)

        print("validation accuracy for max depth of {0}: {1}".format(i, valid_acc))

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

    plot(train_accs, valid_accs)

if __name__ == '__main__':
    main()