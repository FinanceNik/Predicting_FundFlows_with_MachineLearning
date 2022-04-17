import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


def dataset_combiner():
    completed_dataset = []
    path = '/home/thequant/Downloads/'
    for i in range(1,8):
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

    description = df.describe(percentiles=[]).transpose().round(4)
    print(description)


# time_series_data_description()


def fund_characteristics_data_description():
    df = pd.read_csv('data/Morningstar_data_version_1.1_filtered.csv')
    df.drop(list(df.filter(regex='Annual Report Net Expense Ratio')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Monthly Gross Return')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Estimated Share Class Net Flow')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    description = df.describe(percentiles=[]).transpose().round(4)
    print(description)


# fund_characteristics_data_description()


def data_description():
    df = pd.read_csv('data/Morningstar_data_version_1.1_filtered.csv')
    print(df.info())
    # print(df.columns[:])

    # --> Create a function that describes the fund characteristics. How much data is missing? Visualize that!
    # df.drop([all fund flows, return and other time series variables.], axis=1)

    # --> Create a function with df.describe() && df.info() so that the reader of the thesis can have a look
    # at the data in its meta form.

    # --> Decide on which columns / characteristics to drop based on the amount of missing information!
    # ----> Or interpolate the data with average / mean / null / -9999 values?



