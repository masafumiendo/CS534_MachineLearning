#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

class PreProcess:

    # Public method
    def get_data(self):

        # Read .csv data
        df_train = pd.read_csv('pa3_train.csv')
        df_valid = pd.read_csv('pa3_val.csv')
        df_test = pd.read_csv('pa3_test.csv')

        # change column names to remove '-' from col names because of pandas issue, also class-> Class bc of class object, replace '?'
        df_train.columns = [col.replace("-", "").replace("class", "Class").replace("?", "unk") for col in df_train.columns]
        df_valid.columns = [col.replace("-", "").replace("class", "Class").replace("?", "unk") for col in df_valid.columns]
        df_test.columns = [col.replace("-", "").replace("?", "unk") for col in df_test.columns]

        # remove columns where value is same for all molecules - ex: veil-type_p
        nunique = df_train.apply(pd.Series.nunique)
        cols_to_drop = list(nunique[nunique == 1].index)

        df_train = df_train.drop(cols_to_drop, axis=1)
        df_valid = df_valid.drop(cols_to_drop, axis=1)
        df_test = df_test.drop(cols_to_drop, axis=1)

        return df_train, df_valid, df_test