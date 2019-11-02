"""
Author: Masafumi Endo, Dilan Senaratne, Morgan Mayer
Date: 10/29
Python Version: 3.6
Objective: Implementation of online perceptron for part 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KernalPerceptron:

    # Constructor
    def __init__(self, x_train, y_train, x_valid, y_valid, iters):

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.iters = iters
        self.num_sample_train = len(x_train)
        self.num_sample_valid = len(x_valid)

        # self.alpha = np.zeros(self.num_sample_train)

        self.accuracy_train = np.zeros(iters)
        self.accuracy_valid = np.zeros(iters)

    # Public method
    # Method for training procedure
    def train(self, p, figname):

        alpha_list = []
        alpha = np.zeros(self.num_sample_train)
        gram_matrix_train_train = self.__grammatrix(self.x_train, self.x_train, p)
        gram_matrix_train_validat = self.__grammatrix(self.x_train, self.x_valid, p)

        # Loop for iterations
        for _iter in range(self.iters):
            # Loop for the number of samples in training data

            for i in range(self.num_sample_train):
                # Compute u
                u = self.__uvalue(alpha, gram_matrix_train_train, self.y_train, i)

                # If function for updating alpha
                if self.y_train[i] * u <= 0:
                    alpha[i] += 1

            _accuracy_train = self.__accuracy_func(self.num_sample_train, alpha, gram_matrix_train_train, self.y_train, self.y_train)
            _accuracy_valid = self.__accuracy_func(self.num_sample_valid, alpha, gram_matrix_train_validat, self.y_train, self.y_valid)
            self.accuracy_train[_iter] = _accuracy_train
            self.accuracy_valid[_iter] = _accuracy_valid

            alpha_iter = alpha
            alpha_list.append(alpha_iter)

        max_accuracy = self.accuracy_valid.max()
        max_accuracy_index = self.accuracy_valid.argmax()
        max_alpha = alpha_list[max_accuracy_index]

        self.__error_graph(figname, option="both")

        return max_accuracy, max_accuracy_index, max_alpha

    # Method for prediction
    def predict(self, x, alpha, p):

        y_test = np.zeros(x.shape[0])
        gram_matrix_train_test = self.__grammatrix(self.x_train, x, p)
        for t in range(x.shape[0]):
            # Compute u to obtain y_hat
            u = self.__uvalue(alpha , gram_matrix_train_test, self.y_train, t)
            y_hat = np.sign(u)

            y_test[t] = y_hat

        return y_test

    # Compute Kernal
    def __kernal(self, x_i, x_j, p):
        kp_xij = (1 + np.dot(x_i, x_j))**p
        return kp_xij

    # Compute Gram matrix
    def __grammatrix(self, x1, x2, p):
        # gram_matrix = np.zeros((x1.shape[0], x2.shape[0]))
        # for i in range(x1.shape[0]):
        #    for j in range(x2.shape[0]):
        #        gram_matrix[i,j] = self.__kernal(x1[i,:], x2[j,:], p)

        gram_matrix = (1 + np.matmul(x1, np.transpose(x2)))**p
        # diff = gram_matrix2 - gram_matrix
        return gram_matrix

    # Calculate u values
    def __uvalue(self, alpha, gram_matrix, y, t):
        u = 0
        for i in range(self.num_sample_train):
            u += alpha[i]*gram_matrix[i, t]*y[i]
        return u

    # Method for calculate accuracy
    def __accuracy_func(self, num_sample, alpha, gram_matrix, y_train, y):

        L = 0.

        for t in range(num_sample):
            u = self.__uvalue(alpha, gram_matrix, y_train, t)
            if y[t]*u > 0:
                L += 1

        L = (L / num_sample)

        return L

    # Method for drawing error graph
    def __error_graph(self, figname, option):

        if option == "both":

            fig = plt.figure()
            plt.plot(np.arange(0, len(self.accuracy_train)), self.accuracy_train, label="training accuracy")
            plt.plot(np.arange(0, len(self.accuracy_valid)), self.accuracy_valid, label="validation accuracy")

            plt.xlabel("iteration")
            plt.ylabel("accuracy")
            # plt.ylim((0.9, 1.05))

            plt.title("")
            plt.legend()

            plt.savefig('fig/part3/' + figname)
            plt.close(fig)

        elif option == "separate":

            fig = plt.figure()
            plt.plot(np.arange(0, len(self.accuracy_train)), self.accuracy_train, label="training accuracy")
            plt.xlabel("iteration")
            plt.ylabel("accuracy")
            plt.ylim((0.9, 1.0))

            plt.title("")
            plt.legend()

            plt.savefig('fig/part3/training_' + figname)
            plt.close(fig)

            fig = plt.figure()
            plt.plot(np.arange(0, len(self.accuracy_valid)), self.accuracy_valid, label="validation accuracy")
            plt.xlabel("iteration")
            plt.ylabel("accuracy")
            plt.ylim((0.5, 1.0))

            plt.title("")
            plt.legend()

            plt.savefig('fig/part3/validation_' + figname)
            plt.close(fig)

    def p_graph(self, max_accuracy_list, p_mat, figname):

        fig = plt.figure()
        plt.plot(np.arange(1, len(p_mat)+1), max_accuracy_list, label="accuracy vs p")
        plt.xlabel("p values")
        plt.ylabel("best accuracy")
        plt.title("")
        plt.legend()

        plt.savefig('fig/part3/' + figname)
        plt.close(fig)

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


    p_mat = [1, 2, 3, 4, 5]
    # p_mat = [1, 2]
    max_accuracy_list = np.zeros(len(p_mat))
    max_itr_list = np.zeros(len(p_mat))
    alpha_list = []

    kernalperceptron = KernalPerceptron(x_train, y_train, x_valid, y_valid, iters=15)

    for i in range(len(p_mat)):
        p = p_mat[i]
        max_accuracy_list[i], max_itr_list[i], alpha = kernalperceptron.train(p, 'accuracy_graph_p{}.png'.format(p))
        alpha_list.append(alpha)

    max_accuracy_p = max_accuracy_list.max()            # Maximum accuracy in all p
    max_accuracy_p_index = max_accuracy_list.argmax()

    max_p = p_mat[max_accuracy_p_index]                 # Best p value
    max_itr_p = max_itr_list[max_accuracy_p_index]      # best iteration
    max_alpha = alpha_list[max_accuracy_p_index]        # best alpha


    y_test = kernalperceptron.predict(x_test, max_alpha, max_p)
    y_test = pd.DataFrame(y_test)
    y_test.to_csv('kplabel.csv',header=False, index=False)
    summary = pd.Series({'best accuracy': max_accuracy_list, 'best iteration': max_itr_list, 'best p': max_p, 'best alpha': max_alpha})

    kernalperceptron.p_graph(max_accuracy_list, p_mat, "Accuracy_p")
    summary.to_csv('summary_kplabel.csv')

if __name__ == '__main__':
    main()
