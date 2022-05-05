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

    df.to_csv('data/Morningstar_data_version_2.0.csv')

    return df


def remove_many_nans():
    df = pd.read_csv('data/Morningstar_data_version_2.0.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    col_drops = ['Time Horizon', '# of \nStock \nHoldings (Short)',
                 'Average Market Cap \n(mil) (Short) \nPortfolio \nCurrency', 'ff_data_points', 'Performance Fee',
                 'Beta \n2000-01-01 \nto 2022-03-31 \nBase \nCurrency',
                 'Estimated Share Class Net Flow (Monthly) \n2022-01 \nBase \nCurrency',
                 'Estimated Share Class Net Flow (Monthly) \n2022-02 \nBase \nCurrency']
    for col_drop in col_drops:
        df.drop([col_drop], axis=1, inplace=True)

    row_drops = ['Net Assets \n- Average', 'Manager \nTenure \n(Longest)', 'Percent of Female Executives', 'Firm City',
                 'P/E Ratio (TTM) (Long)', 'Investment Area', 'Management \nFee']
    for row_drop in row_drops:
        df = df[df[row_drop].notna()]

    mgmt_fee = df.pop('Management \nFee')
    df.insert(9, 'Management \nFee', mgmt_fee)

    return df


def convert_to_panel_data():
    df = remove_many_nans()
    df = df[:10]

    list_cols = []
    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.
    for year in list_years:
        for month in list_months:
            list_cols.append(f'Estimated Share Class Net Flow (Monthly) \n{year}-{month} \nBase \nCurrency')

    drops = df.loc[:, ~df.columns.isin(list_cols)]
    drops = list(drops.columns[:])

    df3 = pd.melt(frame=df, id_vars=drops, var_name="year-month", value_name='ff')

    year = df3["year-month"].str[42:46].astype(int)
    month = df3["year-month"].str[47:50].astype(int)

    df3.insert(2, 'year', year)
    df3.insert(2, 'month', month)

    # Split the colum year-month into two columns for year & month, then match the year and month to the other time-series
    # data points and fill them in.

    df3.to_csv('ZZZZZZ.csv')


convert_to_panel_data()