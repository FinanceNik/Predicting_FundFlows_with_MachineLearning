import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import random

# helps with debugging purposes. Allows for the terminal to display up to 500 columns/ rows.
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# Sometimes pandas throws a SettingWithCopy warning.  This warning can be ignored as its only there for safety purposes.
pd.options.mode.chained_assignment = None  # default='warn'


# As the version of Morningstar does not allow for the export of the whole dataset at once, it has to be combined.
def dataset_combiner():
    completed_dataset = []
    path = '/home/thequant/Downloads/'
    for i in range(1, 8):
        filename = f'version_1.1_part_{i}.csv'
        df = pd.read_csv(path+filename)
        completed_dataset.append(df)

    completed_dataset = pd.concat(completed_dataset)
    completed_dataset.to_csv('Morningstar_data_version_1.1.csv')
    print(len(completed_dataset.index))


# After going through the data with df.describe(), I was able to see that some columns did not have any data in them.
# These columns can thus be dropped in order to create a more sensible, cleaner dataset.
def usable_dataset_cleaner():

    # Load the raw dataset directly combined off of Morningstar.
    df = pd.read_csv('data/Morningstar_data_version_1.1.csv')

    # The following variables had to be dropped:
    # --> Unnamed: 0 --> Was the index column created by Morningstar when downloading the data. Do not want to use it.
    # --> Alpha 1 Yr... --> After talking to Moreno, he said not to use the Alpha off Morningstar. Downloaded only for
    # testing purposes but should not be included in any analysis.
    # --> Public --> Did not have any information in it.
    # --> Number of Shareholders --> Did not have any information in it.
    # --> Total Expense Ratio --> Did not have any information in it.
    # --> Net Expense Ratio --> Only has one data-point in it.
    df = df.drop(['Unnamed: 0',
                  'Alpha 1 Yr (Gross Return)(Qtr-End)',
                  'Public',
                  'Number of Shareholders',
                  'Net Expense \nRatio',
                  'Total Expense Ratio'], axis=1)

    # The following list of variables had to be dropped additionally:
    # --> All 'Annual Report Management Expense Ratio (MER) Year XY --> Did not have any information in it.
    df.drop(list(df.filter(regex='MER')), axis=1, inplace=True)

    # After further inspection of the data I have realized that the number are not properly formatted natively when
    # exporting the data off of Morningstar. Numerical Values > 999 are delimited by '. This character thus has to
    # be removed.
    df = df.replace('’', '', regex=True)

    # And this is the adjusted raw dataset, exported as .csv for further analysis.
    df.to_csv('data/Morningstar_data_version_1.1_filtered.csv')


# This function convert all the non-numerical values to numerical ones with the only exception being the fund's name.
# This is to be done in order for ML algorithms to work on the data. They typically do not take text-based data.
# The exception to the fund name is chosen as the name can be used as an index later and thus does not conflict with
# quantitative models.
def convert_categorical_data():
    df = pd.read_csv('data/Morningstar_data_version_1.1_filtered.csv')

    def conversion_firmCity():
        unique_cities_df = df.drop_duplicates(subset=['Firm City'])
        unique_cities = unique_cities_df['Firm City'].to_list()
        unique_city_id = random.sample(range(1_000, 50_000), len(unique_cities))
        ID_dict = dict(zip(unique_cities, unique_city_id))
        df['Firm City ID'] = df['Firm City'].map(ID_dict)
        return df

    def conversion_fund_firm():
        df_new = conversion_firmCity()
        unique_fundFirms_df = df_new.drop_duplicates(subset=['Management Company'])
        unique_fundFirms = unique_fundFirms_df['Management Company'].to_list()
        unique_fundFirms_id = random.sample(range(1_000, 50_000), len(unique_fundFirms))
        ID_dict = dict(zip(unique_fundFirms, unique_fundFirms_id))
        df['Management Company ID'] = df['Management Company'].map(ID_dict)
        return df

    def conversion_investment_area():
        df_new = conversion_fund_firm()
        unique_investmentArea_df = df_new.drop_duplicates(subset=['Investment Area'])
        unique_investmentArea = unique_investmentArea_df['Investment Area'].to_list()
        unique_investmentArea_id = random.sample(range(1_000, 50_000), len(unique_investmentArea))
        ID_dict = dict(zip(unique_investmentArea, unique_investmentArea_id))
        df['Investment Area ID'] = df['Investment Area'].map(ID_dict)
        return df

    def conversion_Morningstar_Category():
        df_new = conversion_investment_area()
        unique_investmentArea_df = df_new.drop_duplicates(subset=['Morningstar Category'])
        unique_investmentArea = unique_investmentArea_df['Morningstar Category'].to_list()
        unique_investmentArea_id = random.sample(range(1_000, 50_000), len(unique_investmentArea))
        ID_dict = dict(zip(unique_investmentArea, unique_investmentArea_id))
        df['Morningstar Category ID'] = df['Morningstar Category'].map(ID_dict)
        return df

    df_numerical_only = conversion_Morningstar_Category()
    df_numerical_only = df_numerical_only.drop(['Firm City',
                                                'Management Company',
                                                'Investment Area',
                                                'Morningstar Category'], axis=1)
    df_numerical_only.to_csv('data/Morningstar_data_version_1.1_filtered_numOnly.csv')


# This function describes the variables of the whole dataset.
def raw_data_description():
    df = pd.read_csv('data/Morningstar_data_version_1.1_filtered.csv')
    description = df.describe(percentiles=[]).transpose().round(4)
    # description.to_csv('raw_data_describe.csv')
    print(df.info())
    print('\n')
    print(description)


# This function describes the variables of a dataset consisting of only the columns with time-series-like information.
def time_series_data_description():
    df = pd.read_csv('data/Morningstar_data_version_1.1_filtered.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='IPO')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='P/E')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Net Assets')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='#')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Average')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Manager')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Female')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Firm City ID')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Management Company ID')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Investment Area ID')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Morningstar Category')), axis=1, inplace=True)

    description = df.describe(percentiles=[]).transpose().round(4)
    description.to_csv('time_series_data_describe.csv')


# This function describes the variables of a dataset consisting of only the raw fnd characteristics retrieved by
# Morningstar.
def fund_characteristics_data_description():
    df = pd.read_csv('data/Morningstar_data_version_1.1_filtered_numOnly.csv')
    df.drop(list(df.filter(regex='Annual Report Net Expense Ratio')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Monthly Gross Return')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Estimated Share Class Net Flow')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    description = df.describe(percentiles=[]).transpose().round(4)
    description.to_csv('fund_characteristics_data_describe.csv')


# As described in the paper, the main objective is to analyze fund flows.
def drop_if_not_enough_ff_data():
    df = pd.read_csv('data/Morningstar_data_version_1.1_filtered_numOnly.csv')
    df.insert(4, 'ff_data_points', '')

    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))

    # This loop first iterates over a row in the dataset, then over its year and finally the months in order to
    # Count the number of occurring data-points in the fund flow columns. After that, the number of occurrences is
    # written to a new column based on which the third-to-last line will then delete all funds with less than 36m
    # of data.
    for i in range(len(df.index)):
        ff_data = []
        for year in list_years:
            for month in list_months:
                # This is called an f-string which allows for the insertion of variables directly into the string.
                ff_column = f'Estimated Share Class Net Flow (Monthly) \n{year}-{month} \nBase \nCurrency'
                value = df[ff_column][i]
                # Append the value to the list.
                ff_data.append(value)

        # This is called list-comprehension. It allows for the manipulation of a list within one line of code.
        sum_ff_points = int(sum(x > 0 or x < 0 for x in ff_data))
        # Fill in the number of data points to the column.
        df['ff_data_points'][i] = sum_ff_points
        print(f'Setting ff_data point for: {i} -> done!')  # --> Verbose output to check on the progress.

    # Drop the funds with less than 36 months of data.
    df = df.drop(df[df.ff_data_points < 36].index)
    df.to_csv('data/Morningstar_data_version_1.2_filtered_numOnly.csv')  # --> Save the dataset.
    print('Operation Finalized!')  # --> Verbose output to check if operation done.


# Function that converts all the missing values to zero. Better choice than dropping all the rows in which some
# values are missing.
def fill_NaN_df():
    df = pd.read_csv('data/Morningstar_data_version_1.2_filtered_numOnly.csv')
    df = df.fillna(0.0)
    df.to_csv('data/Morningstar_data_version_1.3_filtered_numOnly.csv')  # --> Save the dataset.


# Function that inserts a row to be populated with the S&P 500 Data off Yahoo in order to calculate alpha later on.
def insert_SandP500_data():
    df = pd.read_csv('data/Morningstar_data_version_1.3_filtered_numOnly.csv')

    fill_data = []
    for i in range(len(df.columns[:])):
        fill_data.append(0.0)
    df.loc[-1] = fill_data  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index
    df['Name'][0] = 'SandP500'

    # Reverse the S&P dataset as its latest to oldest.
    df_snp = pd.read_csv('data/SnP500_data.csv')
    df_snp = df_snp.iloc[::-1]

    df_snp.to_csv('data/SnP500_data.csv')
    df.to_csv('data/Morningstar_data_version_1.4_filtered_numOnly.csv')  # --> Save the dataset.


def insert_risk_free_rate_data(data):
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


