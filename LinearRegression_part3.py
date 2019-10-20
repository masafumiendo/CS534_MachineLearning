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

        # weight = np.random.random_sample(x_train.shape[1])
        weight = np.zeros(x_train.shape[1])
        weight_ref = weight
        norm_gradient_ref = np.array([])

        error_train = self.__calc_error(x_train, y_train, weight)
        error_train_pre = error_train

        epoch = 1
        max_epoch = 100
        count = 0  # To check whether exploding
        flag_div = False

        while True:
            gradient = 0

            for j in range(x_train.shape[0]):
                _x = x_train[j, :]
                _y = y_train[j]
                _z = self.__regression(_x, weight)

                _gradient = - 2 * (_y - _z) * _x
                gradient += _gradient

            # gradient_normalize = gradient / (x_train.shape[0])gradient_normalize
            gradient += 2 * regularization * weight

            # Weight update
            weight = weight - lr * gradient

            # Calculate training error
            _error_train = self.__calc_error(x_train, y_train, weight)  # SSE
            error_train = np.append(error_train, _error_train)

            norm_gradient = np.sqrt(np.dot(gradient, gradient))  # Norm of the gradient
            norm_gradient_ref = np.append(norm_gradient_ref, norm_gradient)
            #print(_error_train)

            # Save weight at each epoch for calculating validation error
            weight_ref = np.append(weight_ref, weight)

            if error_train_pre < _error_train:
                count += 1
                error_train_pre = _error_train
            else:
                count = 0
                error_train_pre = _error_train

            if norm_gradient <= epsilon or epoch > max_epoch or count > 9:
                if count > 9:
                    flag_div = True
                break

            epoch += 1

        weight_ref = np.reshape(weight_ref, (-1, x_train.shape[1]))

        return weight, weight_ref, error_train, epoch, flag_div, norm_gradient_ref

    def valid(self, x_valid, y_valid, weight_ref, epoch):

        error_valid = np.zeros(epoch)

        # Loop for computing error
        for i in range(epoch):
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

    def error_graph(self, figname, error, flag_div, title_name):

        if flag_div:
            fig = plt.figure()
            plt.plot(np.arange(0, error.shape[0]), error)
            # plt.plot(np.arange(0, error_valid.shape[0]), error_valid, label='validation error')
            plt.xlabel('epoch')
            plt.ylabel('SSE')
            # plt.legend()
            plt.yscale('log')
            plt.xscale('log')
            plt.show()
            plt.title(title_name)
            plt.savefig('figure_part1/' + figname)
            plt.close(fig)
        else:
            fig = plt.figure()
            plt.plot(np.arange(0, error.shape[0]), error)
            # plt.plot(np.arange(0, error_valid.shape[0]), error_valid, label='validation error')
            plt.xlabel('epoch')
            plt.ylabel('SSE')
            # plt.legend()
            plt.yscale('log')
            plt.xscale('log')
            plt.show()
            plt.title(title_name)
            plt.savefig('figure_part1/' + figname)
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

    def train(self, path):
        df = pd.read_csv(path)
        x = df.drop(['dummy', 'id'], axis=1)

        x_datetime = pd.to_datetime(x['date'], infer_datetime_format=True)
        x_year, x_month, x_day = x_datetime.dt.year, x_datetime.dt.month, x_datetime.dt.day
        x_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
        x = x.drop(['date'], axis=1)
        x = pd.concat([x_date, x], axis=1)

        x_id = df.loc[:, 'dummy']  # Dummy variable
        x = pd.concat([x_id, x], axis=1)  # Add dummy
        y = x.loc[:, 'price']
        x = x.drop(['price'], axis=1)  # Remove price

        x, y = x.values, y.values

        return x, y

    def valid(self, path):
        df = pd.read_csv(path)
        x = df.drop(['dummy', 'id'], axis=1)

        x_datetime = pd.to_datetime(x['date'], infer_datetime_format=True)
        x_year, x_month, x_day = x_datetime.dt.year, x_datetime.dt.month, x_datetime.dt.day
        x_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
        x = x.drop(['date'], axis=1)
        x = pd.concat([x_date, x], axis=1)

        x_id = df.loc[:, 'dummy']
        x = pd.concat([x_id, x], axis=1)

        y = x.loc[:, 'price']
        x = x.drop(['price'], axis=1)  # Remove price

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

        x_id = df.loc[:, 'dummy']
        x = pd.concat([x_id, x], axis=1)

        x = x.values

        return x


def main():
    feature_eng = FeatureEngineering()

    x_train, y_train = feature_eng.train('C:\Fall 2019\CS534_MachineLearning\PA1_train.csv')
    x_valid, y_valid = feature_eng.valid('C:\Fall 2019\CS534_MachineLearning\PA1_dev.csv')
    x_test = feature_eng.predict('C:\Fall 2019\CS534_MachineLearning\PA1_test.csv')

    regularization = 0
    epsilon = 0.5
    # lr_mat = [1, 0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    lr_mat = [1]

    for j in range(len(lr_mat)):
        lr = lr_mat[j]
        linear_regression = LinearRegression()

        weight, weight_ref, error_train, epoch, flag_div, norm_gradient_ref = linear_regression.train(x_train, y_train, lr, regularization,
                                                                                   epsilon)
        error_valid = linear_regression.valid(x_valid, y_valid, weight_ref, epoch)

        # linear_regression.error_graph('error_train_lr{}.png'.format(lr), error_train, flag_div, 'Training SSE')
        # linear_regression.error_graph('error_valid_lr{}.png'.format(lr), error_valid, flag_div, 'Validating SSE')
        linear_regression.error_graph('gradient_valid_lr{}.png'.format(lr), norm_gradient_ref, flag_div, 'L2 norm Gradient')

        y_test = linear_regression.predict(x_test, weight)


if __name__ == '__main__':
    main()
