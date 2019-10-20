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
        #weight = np.zeros(x_train.shape[1])
        weight_ref = weight

        error_train = self.__calc_error(x_train, y_train, weight)
        error_train_normalize = error_train / x_train.shape[0]
        error_train_pre = error_train

        epoch = 1
        max_epoch = 1000
        count = 0  # To check whether exploding

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
            _error_train = self.__calc_error(x_train, y_train, weight)          # SSE
            _error_train_normalize = _error_train / x_train.shape[0]
            error_train_normalize = np.append(error_train_normalize, _error_train_normalize)

            norm_gradient = np.sqrt(np.dot(gradient, gradient))  # Norm of the gradient
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
                break

            epoch += 1

        weight_ref = np.reshape(weight_ref, (-1, x_train.shape[1]))

        return weight, weight_ref, error_train_normalize, epoch

    def valid(self, x_valid, y_valid, weight_ref, epoch):

        error_valid = np.zeros(epoch)

        # Loop for computing error
        for i in range(epoch):
            _weight = weight_ref[i, :]
            error_valid[i] = self.__calc_error(x_valid, y_valid, _weight)

        error_valid_normalize = error_valid # / x_valid.shape[0]
        return error_valid_normalize

    def predict(self, x_test, weight, x_min, x_max):

        num = x_test.shape[0]
        y_test = np.zeros(num)

        for i in range(num):
            y_test[i] = self.__regression(x_test[i, :], weight)
            #y_test[i] = (x_max - x_min) * y_test[i] + x_min  # Reverse the normalization
            print("Predicted cost is $", y_test[i], ".")

        return y_test

    def error_graph(self, figname, error_train, error_valid):

        fig = plt.figure()
        plt.plot(np.arange(0, error_train.shape[0]), error_train, label='training error')
        plt.plot(np.arange(0, error_valid.shape[0]), error_valid, label='validation error')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        #plt.yscale('log')
        #plt.xscale('log')
        #plt.show()
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

    def plot_sse_epoch(self, dict_sse):
        fig = plt.figure()
        for _lr in dict_sse:
            print(_lr, len(dict_sse[lr]))
            plt.plot(dict_sse[_lr], label=str(_lr))
        plt.xlabel('epoch')
        plt.ylabel('SSE')
        plt.legend()
        #plt.yscale('log')
        #plt.xscale('log')
        #plt.show()
        plt.savefig('figure_part1/sse_epoch_lr.png')
        plt.close(fig)

    def percent_diff(self, y_actual, y_pred):
        percent_diff = 100*(np.abs(y_actual - y_pred)) / y_actual #absolute percent difference
        return percent_diff
        
    def plot_box_lr(self, dict_error):
        fig = plt.figure()
        
        # or backwards compatable    
        labels, data = dict_error.keys(), dict_error.values()
        
        plt.boxplot(data, showfliers=False)
        plt.xticks(range(1, len(labels) + 1), labels)        
        plt.xlabel('Learning Rate')
        plt.ylabel('Percent difference in price prediction')
        #plt.show()
        plt.savefig('figure_part1/percent_diff_boxplot.png')
        plt.close(fig)


class FeatureEngineering:

    def train(self, path):
        df = pd.read_csv(path)
        x = df.drop(['dummy', 'id'], axis=1)

        x_datetime = pd.to_datetime(x['date'], infer_datetime_format=True)
        x_year, x_month, x_day = x_datetime.dt.year, x_datetime.dt.month, x_datetime.dt.day
        x_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
        x = x.drop(['date'], axis=1)
        x = pd.concat([x_date, x], axis=1)

        x_min = x.min()
        x_max = x.max()

        x_normalized = (x - x_min) / (x_max - x_min)

        x_id = df.loc[:, 'dummy']  # Dummy variable
        x_normalized = pd.concat([x_id, x_normalized], axis=1)  # Add dummy
        y = x_normalized.loc[:, 'price']
        x_normalized = x_normalized.drop(['price'], axis=1)  # Remove price

        x, y = x_normalized.values, y.values

        return x, y, x_min, x_max

    def valid(self, path, x_min, x_max):
        df = pd.read_csv(path)
        x = df.drop(['dummy', 'id'], axis=1)

        x_datetime = pd.to_datetime(x['date'], infer_datetime_format=True)
        x_year, x_month, x_day = x_datetime.dt.year, x_datetime.dt.month, x_datetime.dt.day
        x_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
        x = x.drop(['date'], axis=1)
        x = pd.concat([x_date, x], axis=1)

        x_normalized = (x - x_min) / (x_max - x_min)

        x_id = df.loc[:, 'dummy']
        x_normalized = pd.concat([x_id, x_normalized], axis=1)

        y = x_normalized.loc[:, 'price']
        x_normalized = x_normalized.drop(['price'], axis=1)  # Remove price

        x, y = x_normalized.values, y.values

        return x, y

    def predict(self, path, x_min, x_max):
        x_min = x_min.drop(['price'])
        x_max = x_max.drop(['price'])

        df = pd.read_csv(path)
        x = df.drop(['dummy', 'id'], axis=1)

        x_datetime = pd.to_datetime(x['date'], infer_datetime_format=True)
        x_year, x_month, x_day = x_datetime.dt.year, x_datetime.dt.month, x_datetime.dt.day
        x_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
        x = x.drop(['date'], axis=1)
        x = pd.concat([x_date, x], axis=1)

        x_normalized = (x - x_min) / (x_max - x_min)
        x_id = df.loc[:, 'dummy']
        x_normalized = pd.concat([x_id, x_normalized], axis=1)

        x = x_normalized.values

        return x

#def main():



if __name__ == '__main__':
    #main()

    feature_eng = FeatureEngineering()

#    x_train, y_train, x_min, x_max = feature_eng.train('C:\Fall 2019\CS534_MachineLearning\PA1_train.csv')
#    x_valid, y_valid = feature_eng.valid('C:\Fall 2019\CS534_MachineLearning\PA1_dev.csv', x_min, x_max)
#    x_test = feature_eng.predict('C:\Fall 2019\CS534_MachineLearning\PA1_test.csv', x_min, x_max)

    x_train, y_train, x_min, x_max = feature_eng.train('PA1_train.csv')
    x_valid, y_valid = feature_eng.valid('PA1_dev.csv', x_min, x_max)
    x_test = feature_eng.predict('PA1_test.csv', x_min, x_max)


    regularization = 0
    epsilon = 0.5
    #lr_mat = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    lr_mat = [1e-5, 1e-6, 1e-7]
    dict_sse = {} 
    dict_pred = {}

    for j in range(len(lr_mat)):
        lr = lr_mat[j]
        linear_regression = LinearRegression()

        weight, weight_ref, error_train_normalize, epoch = linear_regression.train(x_train, y_train, lr, regularization, epsilon)
        error_valid_normalize = linear_regression.valid(x_valid, y_valid, weight_ref, epoch)

        linear_regression.error_graph('error_lr{}.png'.format(lr), error_train_normalize, error_valid_normalize)
        y_test = linear_regression.predict(x_test, weight, x_min.loc['price'], x_max.loc['price'])
        y_pred = linear_regression.predict(x_valid, weight, x_min.loc['price'], x_max.loc['price'])
        
        diff = linear_regression.percent_diff(y_valid, y_pred)
        dict_pred[lr] = diff

        dict_sse[lr] = error_valid_normalize
        
    linear_regression.plot_sse_epoch(dict_sse)

    linear_regression.plot_box_lr(dict_pred)
        