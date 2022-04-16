import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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


def dataset_cleaner():
    df = pd.read_csv('data/Morningstar_data_version_1.1.csv')
    df = df.drop(['Unnamed: 0'], axis=1)

    # --> First I have to clean up the columns, 600 is way too much to visualize.
    # sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    # plt.show()
    # print(df.describe())


dataset_cleaner()


#######################################################################################################
# --> Functions below are only for testing purposes and do not contribute to the final analysis.
#######################################################################################################


def data_cleaning():
    df = pd.read_csv('data/MutualFund prices - A-E.csv')
    df['price_date'] = pd.to_datetime(df['price_date'], format="%Y-%m-%d")

    df = df.loc[(df['fund_symbol'] == 'AAAAX')]
    df.insert(3, 'period_change_value', '')
    df['period_change_value'] = df['nav_per_share'].pct_change()
    df.insert(4, 'period_change_positive', '')

    def period_change_calculator(x):
        if x >= 0.0:
            return 1
        else:
            return 0

    df['period_change_positive'] = df['period_change_value'].apply(period_change_calculator)
    df.insert(5, 'net_fund_flow', '')
    df['net_fund_flow'] = df['nav_per_share'].diff(periods=1)
    df = df.drop(['fund_symbol'], axis=1)
    df = df.drop(['period_change_value'], axis=1)
    df = df.drop(['period_change_positive'], axis=1)
    df = df.set_index('price_date')
    df = df.fillna(0)

    return df


def data_filtering():
    df = pd.read_csv('data/AAPL.csv')
    df = df.set_index('Date')
    return df


# data_filtering()