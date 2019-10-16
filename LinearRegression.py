"""
Author: Masafumi Endo
Date: 2019/10/13
Version: Python 3.6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:

    # Public method
    def train(self, x_train, y_train, lr, regularization, epsilon):

        weight = np.random.random_sample(x_train.shape[1])
        error_train = np.array([])
        epoch = 0

        while True:

            gradient = 0

            for j in range(x_train.shape[0]):

                _x = x_train[j, :]
                _y = y_train[j]
                _z = self.__regression(_x, weight)
                
                _gradient = - 2 * (_y - _z) * _x + 2 * regularization * weight
                gradient += _gradient
                
            gradient_normalize = gradient / (x_train.shape[0])

            # Weight update
            weight = weight - lr * gradient_normalize
            
            # Save weight at each epoch for calculating validation error
            if epoch == 0:
                weight_ref = weight
            else:
                weight_ref = np.append(weight_ref, weight)

            # Calculate training error
            _error_train = self.__calc_error(x_train, y_train, weight)
            error_train = np.append(error_train, _error_train)

            norm_gradient = np.sqrt(np.dot(gradient_normalize, gradient_normalize))

            if norm_gradient <= epsilon:
                break

            epoch += 1

        weight_ref = np.reshape(weight_ref, (-1, x_train.shape[1]))

        return weight, weight_ref, error_train, epoch

    def valid(self, x_valid, y_valid, weight_ref, epoch):

        error_valid = np.zeros(epoch)

        # Loop for computing error
        for i in range(epoch):

            p = np.random.permutation(len(x_valid))
            _weight = weight_ref[i, :]

            error_valid[i] = self.__calc_error(x_valid, y_valid, _weight)

        return error_valid

    def predict(self, x_test, weight):

        num = x_test.shape[0]
        y_test = np.zeros(num)

        for i in range(num):

            y_test[i] = self.__regression(x_test[i, :], weight)
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

    def __calc_error(self, x, y, weight):

        num = x.shape[0]
        error = 0.
        
        # Loop for calculating error at each target
        for i in range(num):

            _x = x[i, :]
            _y = y[i]

            _z = self.__regression(_x, weight)
            error += (_y - _z) ** 2

        return error

class FeatureEngineering:

    def train_valid(self, path):

        df = pd.read_csv(path)
        x = df.drop(['dummy', 'id', 'price', 'waterfront', 'view', 'zipcode'], axis=1)
        y = df.loc[:, 'price']

        x_datetime = pd.to_datetime(x['date'], infer_datetime_format=True)
        x_year, x_month, x_day = x_datetime.dt.year, x_datetime.dt.month, x_datetime.dt.day
        x_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
        x = x.drop(['date'], axis=1)
        x = pd.concat([x_date, x], axis=1)

        x_normalized = (x - x.min()) / (x.max() - x.min())

        x, y = x_normalized.values, y.values

        return x, y

    def predict(self, path):

        df = pd.read_csv(path)
        x = df.drop(['dummy', 'id', 'waterfront', 'view', 'zipcode'], axis=1)

        x_datetime = pd.to_datetime(x['date'], infer_datetime_format=True)
        x_year, x_month, x_day = x_datetime.dt.year, x_datetime.dt.month, x_datetime.dt.day
        x_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
        x = x.drop(['date'], axis=1)
        x = pd.concat([x_date, x], axis=1)

        x_normalized = (x - x.min()) / (x.max() - x.min())

        x = x_normalized.values

        return x

def main():

    feature_eng = FeatureEngineering()

    x_train, y_train = feature_eng.train_valid('../data/PA1_train.csv')
    x_valid, y_valid = feature_eng.train_valid('../data/PA1_dev.csv')
    x_test = feature_eng.predict('../data/PA1_test.csv')

    lr = 0.1
    regularization = 0.1
    epsilon = 0.5

    linear_regression = LinearRegression()

    weight, weight_ref, error_train, epoch = linear_regression.train(x_train, y_train, lr, regularization, epsilon)
    error_valid = linear_regression.valid(x_valid, y_valid, weight_ref, epoch)

    linear_regression.error_graph('error.png', error_train, error_valid)

    y_test = linear_regression.predict(x_test, weight)

if __name__ == '__main__':
    main()
