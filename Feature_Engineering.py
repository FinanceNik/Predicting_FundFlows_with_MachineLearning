import numpy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('data/Morningstar_data_version_1.9_filtered_numOnly.csv')
# For some reason beyond me, pandas is trying to re-set a new index column every time that df is instantiated.
# Because these are always named 'Unnamed: {}', one can easily filter them with a regular expression.
df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)


# The first feature we would like to create is a fund's monthly alpha value.
# For that we have to calculate the excess return, formula below.
# --> Excess return = RF + β(MR – RF) – TR
def calculate_TR():
    # print(df.columns[50:60])

    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.

    try:
        df.insert(5, 'total_return', '')
    except:
        pass

    for i in range(len(df.index)):
        gross_returns = []
        for year in list_years:
            for month in list_months:
                mgr_column = f'Monthly Gross Return \n{year}-{month} \nBase \nCurrency'
                value = df[mgr_column][i]
                gross_returns.append(value)

        init_value = 1.0
        return_values = []
        for j in range(len(gross_returns)):
            if j == 0:
                return_values.append(init_value)
            else:
                next_value = float(return_values[-1] * (1 + float((gross_returns[j]) / 100)))
                return_values.append(next_value)
        total_return = return_values[-1]
        df['total_return'][i] = total_return

    df.to_csv('data/Morningstar_data_version_1.9_filtered_numOnly.csv')


# --> Excess return = RF + β(MR – RF) – TR
def excess_return(df):
    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.

    # Create the columns for the log return for each month.
    for year in list_years:
        for month in list_months:
            df.insert(1, f'Monthly Excess Return {year} {month}', '')

    for i in range(len(df.index[1:10])):
        gross_returns = []
        for year in list_years:
            for month in list_months:
                mgr_column = f'Monthly Gross Return \n{year}-{month} \nBase \nCurrency'
                value = df[mgr_column][i]
                gross_returns.append(value)

        mean_return = sum(gross_returns) / len(gross_returns)

        for year in list_years:
            for month in list_months:
                mgr_column = f'Monthly Gross Return \n{year}-{month} \nBase \nCurrency'
                mer_column = f'Monthly Excess Return {year} {month}'

                df[mer_column][i] = df[mgr_column][i] - mean_return
                print(df[mer_column][i])



# Notes:
# --> Next to do: Create the function that calculates alpha

# --> Lastly, rebalance the dataset so that the first occurrence of ff data is t=1, next is t=2, ..., t=n