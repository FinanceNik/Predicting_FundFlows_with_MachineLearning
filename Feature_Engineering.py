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

df = pd.read_csv('data/Morningstar_data_version_1.5_filtered_numOnly.csv')
# For some reason beyond me, pandas is trying to re-set a new index column every time that df is instantiated.
# Because these are always named 'Unnamed: {}', one can easily filter them with a regular expression.
df = df.drop(list(df.filter(regex='Unnamed')), axis=1)


# The first feature we would like to create is a fund's monthly alpha value.
# For that we have to calculate the excess return, formula below.
# --> Excess return = RF + β(MR – RF) – TR
def rf_rate_conversion(data):
    rf = pd.read_csv('data/rf.csv')
    rf.rename(columns={'DTB3': 'risk_free',
                       'DATE': 'date'}, inplace=True)
    year = rf['date'].str[:4].astype(int)
    month = rf['date'].str[5:7].astype(int)
    rf.insert(1, "year", year)
    rf.insert(2, "month", month)

    rf.drop(rf.index[:11800], axis=0, inplace=True)
    rf = rf.reset_index()

    def convert_to_float():
        for i in range(len(rf.index)):
            try:
                rf['risk_free'][i] = float(rf['risk_free'][i])
            except:
                rf['risk_free'][i] = 0.0

    convert_to_float()

    fill_data = []
    for i in range(len(data.columns[:])):
        fill_data.append(0.0)
    data.loc[-1] = fill_data  # adding a row
    data.index = data.index + 1  # shifting index
    data = data.sort_index()  # sorting by index
    data['Name'][0] = 'Risk_Free_Rate'

    for year in range(2000, 2022):
        for month in range(1, 13):
            if month <= 9:
                month_str = f'0{month}'
            else:
                month_str = str(month)
            value = round(rf.loc[(rf['year'] == year) & (rf['month'] == month), 'risk_free'].mean(), 5)

            ff_column = f'Monthly Gross Return \n{year}-{month_str} \nBase \nCurrency'
            data[ff_column][0] = value

    data.to_csv('data/Morningstar_data_version_1.6_filtered_numOnly.csv')


rf_rate_conversion(df)


def excess_return(df):
    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.

    # Create the columns for the log return for each month.
    for year in list_years:
        for month in list_months:
            df.insert(1, f'Monthly Excess Return {year} {month}', '')

    for i in range(1):
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



# excess_return(df)




# Notes:
# --> Next to do: Create the function that calculates alpha

# --> Lastly, rebalance the dataset so that the first occurrence of ff data is t=1, next is t=2, ..., t=n