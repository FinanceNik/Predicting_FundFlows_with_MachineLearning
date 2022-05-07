import numpy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import datetime as dt
import DataSet_Cleaner as dsc
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def get_fama_french_data():
    import pandas_datareader.data as reader

    def transform_return_df():
        # Take the df after the dummy variables and then drop all columns except for
        # the name and all the return columns.
        df_transposed = dsc.remove_many_nans()
        df_transposed.drop(list(df_transposed.filter(regex='Unnamed')), axis=1, inplace=True)

        list_cols = ['Name']
        list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.
        for year in list_years:
            for month in list_months:
                list_cols.append(f'Monthly Gross Return \n{year}-{month} \nBase \nCurrency')

        df_transposed = df_transposed[list_cols]
        # fill all nans with 0.0
        df_transposed = df_transposed.fillna(0.0)
        df_transposed = df_transposed.set_index('Name').T

        return df_transposed

    start = dt.date(2000, 1, 1)
    end = dt.date(2021, 12, 31)

    df_fama = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0].reset_index()
    df = transform_return_df().reset_index()

    # The index fucking hates me and I dont know to get rid of this fucking unique index error.
    # --> First guess would be to change one index to a unique value (like random nbrs) and then reset the index

    df_fama.index = list(df_fama.index)
    df.index = list(df.index)

    print(df_fama.index[:20])
    print(df.index[:20])

    df = pd.concat([df, df_fama])
    #
    # df.to_csv('Alpha_Calculation_Dataset.csv')

    return df


get_fama_french_data()


def calculate_alpha():
    df = pd.read_csv('data/Morningstar_data_version_3.0.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)


# calculate_alpha()


# Notes:
# --> Lastly, rebalance the dataset so that the first occurrence of ff data is t=1, next is t=2, ..., t=n