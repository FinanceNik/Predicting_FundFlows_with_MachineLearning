import datetime as dt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import LabelBinarizer
import warnings
# helps with debugging purposes. Allows for the terminal to display up to 500 columns/ rows.
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# Sometimes pandas throws a SettingWithCopy warning.  This warning can be ignored as its only there for safety purposes.
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


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


def convert_return_data():
    df = remove_many_nans()

    list_cols = ['Name']
    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.
    for year in list_years:
        for month in list_months:
            list_cols.append(f'Monthly Gross Return \n{year}-{month} \nBase \nCurrency')

    df_returns = pd.melt(frame=df[list_cols], id_vars=['Name'], var_name="year-month", value_name='monthly_return')

    year = df_returns["year-month"].str[22:26].astype(int)
    month = df_returns["year-month"].str[27:29].astype(int)

    df_returns.insert(2, 'year', year)
    df_returns.insert(2, 'month', month)

    # df_returns.insert(1, 'fund_id', '')
    #
    # for i in range(len(df_returns.index)):
    #     fund_id = f"{df_returns['Name'][i]}_{df_returns['year'][i]}_{df_returns['month'][i]}"
    #     df_returns['fund_id'][i] = fund_id
    #     print(f'done with {i} of {len(df_returns.index)} --> 1')

    df_returns.drop(['year-month'], axis=1, inplace=True)

    return df_returns


def convert_to_panel_data():
    df_returns = convert_return_data()
    df = remove_many_nans()

    list_cols = []
    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.
    for year in list_years:
        for month in list_months:
            list_cols.append(f'Estimated Share Class Net Flow (Monthly) \n{year}-{month} \nBase \nCurrency')

    drops = df.loc[:, ~df.columns.isin(list_cols)]
    drops = list(drops.columns[:])

    df_panel = pd.melt(frame=df, id_vars=drops, var_name="year-month", value_name='fund_flow')

    year = df_panel["year-month"].str[42:46].astype(int)
    month = df_panel["year-month"].str[47:50].astype(int)

    df_panel.insert(2, 'year', year)
    df_panel.insert(2, 'month', month)

    df_panel.drop(list(df.filter(regex='Monthly Gross Return')), axis=1, inplace=True)

    # df_panel.insert(1, 'fund_id', '')

    # for i in range(len(df_panel.index)):
    #     fund_id = f"{df_panel['Name'][i]}_{df_panel['year'][i]}_{df_panel['month'][i]}"
    #     df_panel['fund_id'][i] = fund_id
    #     print(f'done with {i} of {len(df_panel.index)} --> 2')

    monthly_return = df_returns.pop('monthly_return')
    df_panel.insert(40, 'monthly_return', monthly_return)

    df_panel.to_csv('data/Morningstar_data_version_2.1.csv')


def dummy_variables():
    df = pd.read_csv('data/Morningstar_data_version_2.1.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df.drop(['year-month'], axis=1, inplace=True)

    dummy_list = ['Investment Area', 'Morningstar Category', 'Firm City']

    df = pd.get_dummies(df, columns=dummy_list, drop_first=False)

    return df


def convert_annual_expenses():
    df = remove_many_nans()
    df = df.reset_index()
    # df = df[:10]
    # print(df.head())

    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))
    list_cols = ['Name']
    for year in list_years:
        col = f'Annual Report Net Expense Ratio \nYear{year}'
        list_cols.append(col)

    df_annum_exp = df[list_cols]
    df_annum_exp = df_annum_exp.fillna(0.0)
    for year in list_years:
        for month in list_months:
            df_annum_exp.insert(1, f"monthly_exp_ratio_{year}_{month}", "")

    print(len(df_annum_exp.index))
    for i in range(len(df_annum_exp.index)):
        print(f' done with --> {i} | {round((i / len(df_annum_exp.index) * 100), 4)}%')
        for year in list_years:
            annual = df_annum_exp[f'Annual Report Net Expense Ratio \nYear{year}'][i]
            monthly = round(annual / 12, 6)
        for year in list_years:
            for month in list_months:
                df_annum_exp[f"monthly_exp_ratio_{year}_{month}"][i] = monthly

    df_annum_exp.drop(list(df_annum_exp.filter(regex='Annual Report Net Expense Ratio')), axis=1, inplace=True)

    list_cols = ['Name']
    for year in list_years:
        for month in list_months:
            list_cols.append(f'monthly_exp_ratio_{year}_{month}')

    df_exp = pd.melt(frame=df_annum_exp[list_cols], id_vars=['Name'], var_name="year-month", value_name='monthly_exp')

    print(len(df_exp.index))

    df_exp.drop(['year-month'], axis=1, inplace=True)

    df_exp.to_csv('monthly_expenses.csv')


def concat_maindf_and_expdf():
    df = dummy_variables()
    df_exp = pd.read_csv('monthly_expenses.csv')

    monthly_exp = df_exp.pop('monthly_exp')
    df.insert(9, 'monthly_exp', monthly_exp)

    df.drop(list(df.filter(regex='Annual Report Net Expense Ratio')), axis=1, inplace=True)

    df = df.fillna(0.0)

    df.to_csv('data/Morningstar_data_version_3.0.csv')







# Things to do:
# --> What to do with the Management Company Column...cant be dummies, have to do sth else.
# --> Fill all remaining NaN's with 0.0.
# --> Calculate Alpha.
# --> Run the Algos.
# --> Done.
