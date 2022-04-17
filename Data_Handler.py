import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import random
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


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


def raw_data_description():
    df = pd.read_csv('data/Morningstar_data_version_1.1_filtered.csv')
    description = df.describe(percentiles=[]).transpose().round(4)
    # description.to_csv('raw_data_describe.csv')
    print(df.info())
    print('\n')
    print(description)


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


def fund_characteristics_data_description():
    df = pd.read_csv('data/Morningstar_data_version_1.1_filtered_numOnly.csv')
    df.drop(list(df.filter(regex='Annual Report Net Expense Ratio')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Monthly Gross Return')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Estimated Share Class Net Flow')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    description = df.describe(percentiles=[]).transpose().round(4)
    description.to_csv('fund_characteristics_data_describe.csv')


########################################################################
# NOTES:

# --> Create a function that describes the fund characteristics. How much data is missing? Visualize that!
# df.drop([all fund flows, return and other time series variables.], axis=1)

# --> Create a function with df.describe() && df.info() so that the reader of the thesis can have a look
# at the data in its meta form.

# --> Decide on which columns / characteristics to drop based on the amount of missing information!
# ----> Or interpolate the data with average / mean / null / -9999 values?



