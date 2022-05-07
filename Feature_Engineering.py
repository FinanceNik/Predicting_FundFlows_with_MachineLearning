import numpy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def get_fama_french_data():
    import pandas_datareader.data as reader

    start = dt.date(2000, 1, 1)
    end = dt.date(2021, 12, 31)

    factors = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0]
    print(factors)


get_fama_french_data()


def transform_return_df():
    # Take the df after the dummy variables and then drop all columns except for
    # the name and all the return columns.
    # fill all nans with 0.0
    # transform the data such as the format is time series, i.e. funds in cols and
    # returns in rows.
    # insert the fama-french factors into the dataset, at the end. pd.concat([df, df_fama])
    pass


def calculate_alpha():
    df = pd.read_csv('data/Morningstar_data_version_3.0.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)


# calculate_alpha()


# Notes:
# --> Lastly, rebalance the dataset so that the first occurrence of ff data is t=1, next is t=2, ..., t=n