import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', 500)


def data_cleaning():
    df = pd.read_csv('data/MutualFund prices - A-E.csv')
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

    return df


data_cleaning()