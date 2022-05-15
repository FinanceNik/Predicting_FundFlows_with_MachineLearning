import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def data_description():
    df = pd.read_csv('data/Morningstar_data_version_5.0.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    desc = df.describe(percentiles=[]).transpose().round(4)
    desc.to_csv('data/data_description.csv')


def convert_dummies():
    df = pd.read_csv('data/Morningstar_data_version_5.0_lagged.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    def convert_Investment_Area():
        col_type = 'Investment Area'

        df_ia = df.filter(regex=f'{col_type}_')
        df_ia['Name'] = df['Name']
        try:
            df_ia.insert(1, col_type, '')
        except:
            pass

        cols = list(df_ia.filter(regex=col_type).columns[:])
        unique_id = list(range(len(cols)))

        for i in range(len(df_ia.index[:])):
            print(f'loop I --> {round(i / len(df_ia.index[:]), 4)}')
            for j, col in enumerate(cols):
                if df_ia[col][i] == 1:
                    df_ia[col_type][i] = unique_id[j]

        return df_ia

    def convert_Morningstar_Category():
        col_type = 'Morningstar Category'

        df_mc = df.filter(regex=f'{col_type}_')
        df_mc['Name'] = df['Name']
        try:
            df_mc.insert(1, col_type, '')
        except:
            pass

        cols = list(df_mc.filter(regex=col_type).columns[:])
        unique_id = list(range(len(cols)))

        for i in range(len(df_mc.index[:])):
            print(f'loop II --> {round(i / len(df_ia.index[:]), 4)}')
            for j, col in enumerate(cols):
                if df_mc[col][i] == 1:
                    df_mc[col_type][i] = unique_id[j]

        return df_mc

    def convert_Firm_City():
        col_type = 'Firm City'

        df_fc = df.filter(regex=f'{col_type}_')
        df_fc['Name'] = df['Name']
        try:
            df_fc.insert(1, col_type, '')
        except:
            pass

        cols = list(df_fc.filter(regex=col_type).columns[:])
        unique_id = list(range(len(cols)))

        for i in range(len(df_fc.index[:])):
            print(f'loop III --> {round(i / len(df_ia.index[:]), 4)}')
            for j, col in enumerate(cols):
                if df_fc[col][i] == 1:
                    df_fc[col_type][i] = unique_id[j]

        return df_fc

    df_ia = convert_Investment_Area()
    df_mc = convert_Morningstar_Category()
    df_fc = convert_Firm_City()

    Investment_Area = df_ia.pop('Investment Area')
    df.insert(9, 'Investment Area', Investment_Area)

    Morningstar_Category = df_mc.pop('Morningstar Category')
    df.insert(9, 'Morningstar Category', Morningstar_Category)

    Firm_City = df_fc.pop('Firm City')
    df.insert(9, 'Firm City', Firm_City)

    df.drop(list(df.filter(regex='Investment Area_')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Morningstar Category_')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Firm City_')), axis=1, inplace=True)

    df.to_csv('data/Morningstar_data_version_5.0_lagged_noDummies.csv')


def correlation_matrix():
    df = pd.read_csv('data/Morningstar_data_version_5.0_lagged_noDummies.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df = df.rename(columns={'Manager \nTenure \n(Average)': 'Avg. Manager Tenure',
                            'Manager \nTenure \n(Longest)': 'Max. Manager Tenure',
                            'Net Assets \n- Average': 'Avg. Net Assets',
                            'Average Market Cap (mil) (Long) \nPortfolio \nCurrency': 'Avg. Market Cap',
                            'Management \nFee': 'Management Fee'})
    for i, k in enumerate(list(df.columns[:])):
        df = df.rename(columns={list(df.columns[:])[i]: f'{list(df.columns[:])[i]} lagged'})

    df = df.rename(columns={'fund_flow lagged': 'Fund Flow'})

    # print(df.columns[:])

    corr = df.corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.set(rc={'figure.figsize': (11, 10)})
    sns.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    heat = sns.heatmap(corr, mask=mask, annot=True, cmap='Blues', annot_kws={"size": 14})
    heat.set_xticklabels(heat.get_xmajorticklabels(), fontsize=18)
    heat.set_yticklabels(heat.get_ymajorticklabels(), fontsize=18)
    plt.title('Correlation Map of All Variables', fontsize=30)
    heat.figure.axes[-1].yaxis.label.set_size(22)
    plt.tight_layout()
    # plt.show()
    plt.savefig('corr.png', dpi=1000)


# correlation_matrix()