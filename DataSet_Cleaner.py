import datetime as dt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# helps with debugging purposes. Allows for the terminal to display up to 500 columns/ rows.
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# Sometimes pandas throws a SettingWithCopy warning.  This warning can be ignored as its only there for safety purposes.
pd.options.mode.chained_assignment = None  # default='warn'


def data_cleaning():
    df = pd.read_csv('data/Morningstar_data_version_1.1.csv')
    drops = ['Alpha 1 Yr (Gross Return)(Qtr-End)', 'Public', 'Number of Shareholders', 'Net Expense \nRatio',
             'Total Expense Ratio', 'IPO NAV']
    for drop in drops:
        try:
            df.drop([drop], axis=1, inplace=True)
        except:
            print(f'Error dropping: {drop}')

    regex_drops = ['MER', 'Unnamed', 'Morningstar Analyst', 'P/E Ratio (TTM)', 'beta']
    for regex_drop in regex_drops:
        try:
            df.drop(list(df.filter(regex=regex_drop)), axis=1, inplace=True)
        except:
            print(f'Error dropping: {regex_drop}')

    df = df.replace('’', '', regex=True)

    return df


# As described in the paper, the main objective is to analyze fund flows.
def drop_if_not_enough_ff_data():
    df = data_cleaning()
    df.insert(4, 'ff_data_points', '')

    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))

    for year in list_years:
        for month in list_months:
            col = f'Estimated Share Class Net Flow (Monthly) \n{year}-{month} \nBase \nCurrency'
            df[col] = df[col].fillna(0.0)
            df[col] = df[col].astype(float)

    for i in range(len(df.index[:])):
        ff_data = []
        for year in list_years:
            for month in list_months:
                ff_column = f'Estimated Share Class Net Flow (Monthly) \n{year}-{month} \nBase \nCurrency'
                value = df[ff_column][i]
                ff_data.append(value)
        sum_ff_points = int(sum(x > 0 or x < 0 for x in ff_data))
        df['ff_data_points'][i] = sum_ff_points
        print(f'Setting ff_data point for: {i} --> done!')  # --> Verbose output to check on the progress.

    df = df.drop(df[df.ff_data_points < 36].index)

    print(len(df.index))

    return df


def remove_younger_than_3_years():
    df = drop_if_not_enough_ff_data()
    df = df[df['Inception \nDate'].notna()]
    df['Inception \nDate'] = pd.to_datetime(df['Inception \nDate'], format='%Y-%m-%d')

    df = df[~(df['Inception \nDate'] > '2019-04-01')]

    print(len(df.index))

    return df


def remove_many_nans():
    df = remove_younger_than_3_years()
    nans = df.isna().sum()
    nans.to_csv('xx.csv')


remove_many_nans()