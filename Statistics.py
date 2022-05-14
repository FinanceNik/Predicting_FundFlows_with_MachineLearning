import pandas as pd
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


def correlation_matrix():
    df = pd.read_csv('data/Morningstar_data_version_5.0.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    df_ia = df.filter(regex='Investment Area_')
    df_ia['Name'] = df['Name']
    # df_ia.insert(1, 'Investment Area', '')

    cols = list(df_ia.filter(regex='Investment Area').columns[:])
    unique_id = list(range(len(cols))

    for i in range(len(df_ia.index)):
        for col in cols:
            if col == '':



correlation_matrix()
