"""
Author: Masafumi Endo
Date: 2019/10/13
Version: Python 3.6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:

    # Constructor
    def __init__(self, input_dim):

        self.input_dim = input_dim
        self.weight = np.random.random_sample(input_dim)

    # Public method
    def train(self, x_train, y_train, batch_size, lr, regularization, epsilon):

        error_train = np.array([])
        epoch = 0

        while True:

            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]

            for j in range(batch_size):

                _x = x_train[j, :]
                _y = y_train[j]

                _weight_opt, gradient = self.__gradient(_x, _y, lr, regularization)

                if epoch == 0:
                    weight_opt = _weight_opt
                else:
                    weight_opt = np.append(weight_opt, _weight_opt)

            _error_train = self.__calc_error(x_train, y_train, _weight_opt)
            error_train = np.append(error_train, _error_train)

            if gradient <= epsilon:
                break

            epoch += 1

        weight_opt = np.reshape(weight_opt, (-1, self.input_dim))

        return weight_opt, error_train, epoch

    def valid(self, x_valid, y_valid, weight_opt, epoch):

        error_valid = np.zeros(epoch)

        # Loop for computing error
        for i in range(epoch):

            p = np.random.permutation(len(x_valid))
            _x_valid = x_valid[p]
            _y_valid = y_valid[p]
            _weight_opt = weight_opt[i, :]

            error_valid[i] = self.__calc_error(_x_valid, _y_valid, _weight_opt)

        return error_valid

    def predict(self, x_test, weight_opt):

        num = x_test.shape[0]
        y_test = np.zeros(num)

        for i in range(num):

            y_test[i] = self.regression(x_test[i, :], weight_opt[-1])
            print("Predicted cost is $", y_test[i], ".")

        return y_test

    def error_graph(self, figname, error_train, error_valid):

        fig = plt.figure()
        plt.plot(np.arange(0, error_train.shape[0]), error_train, label='training error')
        plt.plot(np.arange(0, error_valid.shape[0]), error_valid, label='validation error')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('../fig/' + figname)
        plt.close(fig)

    def __regression(self, x, weight):

        return np.dot(x, weight)

    def __gradient(self, x, y, lr, regularization):

        z = self.__regression(x, self.weight)
        gradient = - (y - z) * x + (regularization / 2.) * self.weight
        self.weight += lr * gradient

        print(x)
        print(y)
        print(z)
        print(gradient)

        weight_opt = self.weight

        return weight_opt, gradient

    def __calc_error(self, x, y, weight):

        num = x.shape[0]
        error = 0.

        for i in range(num):

            _x = x[i, :]
            _y = y[i]

            _z = self.__regression(_x, weight)
            error += (_y - _z) ** 2 / 2

        return error

class FeatureEngineering:

    def train_valid(self, path):

        df = pd.read_csv(path)
        x = df.drop(['dummy', 'id', 'price'], axis=1)
        y = df.loc[:, 'price']

        x_datetime = pd.to_datetime(x['date'], infer_datetime_format=True)
        x_year, x_month, x_day = x_datetime.dt.year, x_datetime.dt.month, x_datetime.dt.day
        x_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
        x = x.drop(['date'], axis=1)
        x = pd.concat([x_date, x], axis=1)

        x, y = x.values, y.values

        return x, y

    def predict(self, path):

        df = pd.read_csv(path)
        x = df.drop(['dummy', 'id'], axis=1)

        x_datetime = pd.to_datetime(x['date'], infer_datetime_format=True)
        x_year, x_month, x_day = x_datetime.dt.year, x_datetime.dt.month, x_datetime.dt.day
        x_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
        x = x.drop(['date'], axis=1)
        x = pd.concat([x_date, x], axis=1)

        x = x.values

        return x

def main():

    feature_eng = FeatureEngineering()

    x_train, y_train = feature_eng.train_valid('../data/PA1_train.csv')
    x_valid, y_valid = feature_eng.train_valid('../data/PA1_dev.csv')
    x_test = feature_eng.predict('../data/PA1_test.csv')

    batch_size = 1000
    lr = 0.1
    regularization = 0.1
    epsilon = 0.5

    linear_regression = LinearRegression(input_dim=x_train.shape[1])

    weight_opt, error_train, epoch = linear_regression.train(x_train, y_train, batch_size, lr, regularization, epsilon)
    error_valid = linear_regression.valid(x_valid, y_valid, weight_opt, epoch)

    linear_regression.error_graph('error.fig', error_train, error_valid)

    y_test = linear_regression.predict(x_test, weight_opt)

if __name__ == '__main__':
    main()