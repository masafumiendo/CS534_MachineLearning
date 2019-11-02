"""
Author: Masafumi Endo, Dilan Senaratne, Morgan Mayer
Date: 10/29
Python Version: 3.6
Objective: Implementation of online perceptron for part 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

class Perceptron:

    # Constructor
    def __init__(self, x_train, y_train, x_valid, y_valid, iters):

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.iters = iters
        self.num_sample_train = len(x_train)
        self.num_sample_valid = len(x_valid)

        self.weight = np.zeros(x_train.shape[1])
        self.weight_list = []
        
        self.acc_train = np.zeros(iters)
        self.acc_valid = np.zeros(iters)

    # Public method
    # Method for training procedure
    def train(self):

        # Loop for iterations
        for _iter in range(self.iters):

            # Loop for the number of samples in training data
            for i in range(self.num_sample_train):

                # Compute weight dot x
                z = np.dot(self.weight, self.x_train[i, :])
                y_hat = np.sign(z)

                # If function for updating weights
                if self.y_train[i] * z <= 0:
                    self.weight += self.y_train[i] * self.x_train[i, :]
            
            _acc_train = self.__acc(self.num_sample_train, self.x_train, self.y_train)
            _acc_valid = self.__acc(self.num_sample_valid, self.x_valid, self.y_valid)
            self.acc_train[_iter] = _acc_train
            self.acc_valid[_iter] = _acc_valid

            weight_iter = self.weight

            self.weight_list.append(weight_iter)

        self.__error_graph(self.acc_train, self.acc_valid, "acc.png", option="both")

    # Method for prediction
    def predict(self, x):

        max_acc = self.acc_valid.max()
        max_acc_index = self.acc_valid.argmax()

        best_weight = self.weight_list[max_acc_index]

        y_test = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            # Compute weight dot x to obtain y_hat
            z = np.dot(best_weight, x[i, :])
            y_hat = np.sign(z)

            y_test[i] = y_hat

        return y_test, max_acc, max_acc_index
    
    def __acc(self, num_sample, x, y):

        t_cnt = 0

        for i in range(num_sample):

            z = np.dot(self.weight, x[i, :])
            y_hat = np.sign(z)

            if y[i] == y_hat:
                t_cnt += 1

        acc = t_cnt / num_sample

        return acc

    # Method for drawing error graph
    def __error_graph(self, train, valid, figname, option):

        if option == "both":

            fig = plt.figure()
            plt.plot(np.arange(1, len(train)+1), train, label="training accuracy")
            plt.plot(np.arange(1, len(valid)+1), valid, label="validation accuracy")
            plt.xticks(np.arange(1, len(train)+1))
            plt.ylim((0.9, 1.0))
            plt.xlabel("iteration")
            plt.ylabel("accuracy")

            plt.title("")
            plt.legend()

            plt.savefig('fig/part1/' + figname)
            plt.close(fig)

        elif option == "separate":

            fig = plt.figure()
            plt.plot(np.arange(0, len(train)), train, label="training accuracy")

            plt.ylim((0.9, 1.0))

            plt.xlabel("iteration")
            plt.ylabel("loss")

            plt.title("")
            plt.legend()

            plt.savefig('fig/part1/training_' + figname)
            plt.close(fig)

            fig = plt.figure()
            plt.plot(np.arange(0, len(valid)), valid, label="validation accuracy")

            plt.ylim((0.9, 1.0))

            plt.xlabel("iteration")
            plt.ylabel("accuracy")

            plt.title("")
            plt.legend()

            plt.savefig('fig/part1/validation_' + figname)
            plt.close(fig)

    # Method for loss function
    def __loss_func(self, num_sample, x, y):

        L = 0.

        for i in range(num_sample):

            z = np.dot(self.weight, x[i, :])
            L += max(0, - y[i]*z)

        L = L / num_sample

        return L

class Preprocessing:

    # Public method
    # Method for converting labels
    def label_converter(self, y):

        y = y.values

        num_row = y.shape[0]
        y_convert = np.zeros(y.shape[0])

        for i in range(num_row):

            if y[i] == 3:
                y_convert[i] = 1
            elif y[i] == 5:
                y_convert[i] = -1

        return y_convert

    # Method for adding bias term
    def add_bias(self, x):

        bias = pd.DataFrame(np.ones((x.shape[0], 1)))
        x_add_bias = pd.concat([bias, x], axis=1)

        x_add_bias = x_add_bias.values

        return x_add_bias

def main():

    preprocessing = Preprocessing()

    df_train = pd.read_csv('pa2_train.csv')
    df_valid = pd.read_csv('pa2_valid.csv')
    df_test = pd.read_csv('pa2_test_no_label.csv')

    y_train, x_train = df_train.iloc[:, 0], df_train.iloc[:, 1:]
    y_valid, x_valid = df_valid.iloc[:, 0], df_valid.iloc[:, 1:]
    x_test = df_test

    y_train = preprocessing.label_converter(y_train)
    y_valid = preprocessing.label_converter(y_valid)

    x_train = preprocessing.add_bias(x_train)
    x_valid = preprocessing.add_bias(x_valid)
    x_test = preprocessing.add_bias(x_test)

    perceptron = Perceptron(x_train, y_train, x_valid, y_valid, iters=15)
    train = perceptron.train()

    y_test, max_acc, max_acc_index = perceptron.predict(x_test)
    y_test = pd.DataFrame(y_test)
    summary = pd.Series({'best accuracy': max_acc, 'best iteration': max_acc_index+1})

    y_test.to_csv('oplabel.csv', header=False, index=False)
    summary.to_csv('summary_oplabel.csv')

if __name__ == '__main__':
    main()
