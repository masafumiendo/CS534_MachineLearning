import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    
    df = pd.read_csv("PA1_train.csv")
    
    # drop ID feature
    df = df.drop("id", axis=1)
    
    # split date into three columns
    df_datetime = pd.to_datetime(df['date'], infer_datetime_format=True)
    x_year, x_month, x_day = df_datetime.dt.year, df_datetime.dt.month, df_datetime.dt.day
    df_date = pd.DataFrame({'year': x_year, 'month': x_month, 'day': x_day})
    df = df.drop(['date'], axis=1)
    df = pd.concat([df_date, df], axis=1)
    
    # build table that reports mean, std, range for each numerical feature
    categorical = ["year", "month", "day", "waterfront", "view", "condition", "grade", "zipcode",
                   "yr_built", "yr_renovated"]
    numerical = [str(col) for col in df.columns if str(col) not in categorical]
    
    # numerical table of stats
    df_num = pd.DataFrame(index=numerical, columns = ["Mean", "Standard Deviation", "Min", "Max"])
    for n in numerical:
        values = df.eval(n).values
        df_num.loc[str(n), "Mean"] = np.mean(values)
        df_num.loc[str(n), "Standard Deviation"] = np.std(values)
        df_num.loc[str(n), "Min"] = np.min(values)
        df_num.loc[str(n), "Max"] = np.max(values)
        
    # categorical barplots
    df_cat = df.drop(numerical, axis = 1)
    for c in categorical:
        unique = df_cat.eval(c).unique()
        freq = np.array(df_cat[c].value_counts())
        freq = 100 * freq/ np.sum(freq)
        
        if c == "yr_renovated":
            unique = unique[1:]
            freq = freq[1:]
        
        plt.barh(unique, freq)
        plt.xlabel('Percentage')
        plt.ylabel(str(c))
        
        if len(unique) < 30:
            plt.yticks(unique)
            
        plt.savefig("dist/" + str(c) + "_dist.png")
        plt.clf()
        
    # scale data
    feature_eng = FeatureEngineering()
    x_train, y_train, x_min, x_max = feature_eng.train('PA1_train.csv')
    x_valid, y_valid = feature_eng.valid('PA1_dev.csv', x_min, x_max)
    x_test = feature_eng.predict('PA1_test.csv', x_min, x_max)

    
    
    
