import pandas as pd
import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
import DataSet_Cleaner as dsc
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


def count_fund_flow_values():
    df = pd.read_csv('data/Morningstar_data_version_5.0_lagged_noDummies.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    def ff_positive(x):
        if x >= 0.0:
            return 1
        elif x < 0.0:
            return 0

    df['fund_flow'] = df['fund_flow'].apply(ff_positive)
    df = df.rename(columns={'fund_flow': 'Fund Flow'})
    plot = sns.countplot(x='Fund Flow', data=df, palette='Blues')
    plt.title('Fund Flow Occurrences \n (0 = negative ff, 0 = positive ff)', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Observations in mil.', fontsize=22)
    plt.xlabel('Fund Flow Class', fontsize=22)
    plt.show()
    plt.tight_layout()


def distribution_fund_flows():
    df = pd.read_csv('data/Morningstar_data_version_5.0_lagged.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df = df.rename(columns={'fund_flow': 'Fund Flow'})

    df = df[(df["Fund Flow"] > -100_000)]
    df = df[(df["Fund Flow"] < 100_000)]

    sns.histplot(data=df, x='Fund Flow', bins=30000, log_scale=False, kde=True, alpha=0.6)
    plt.ylim(0, 110)
    plt.title('Log. Fund Flow Distribution in USD', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Observations', fontsize=22)
    plt.xlabel('Log. Fund Flow in USD', fontsize=22)
    plt.show()


# distribution_fund_flows()


def average_fund_flow_per_year():
    df = pd.read_csv('data/Morningstar_data_version_5.0_lagged.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df = df[(df["fund_flow"] > 0) | (df["fund_flow"] < 0)]
    df['fund_flow'] = df['fund_flow'] / 1_000_000_000

    list_years = list(range(2000, 2022))
    list_ff = []
    for year in list_years:
        value = int(round(df.loc[df['year'] == year, 'fund_flow'].sum(), 0))
        list_ff.append(value)
    mean = round(sum(list_ff) / len(list_ff), 0)
    print(mean)
    mean_list = [mean for x in range(len(list_ff))]
    plt.grid(alpha=0.6)
    plt.bar(list_years, list_ff, color='#3072a1', alpha=0.99)
    plt.plot(list_years, mean_list, color='#ffa412', alpha=0.99, linewidth=5)
    plt.ylim(-600, 200)
    plt.title('Sum of Fund Flow per Year in bn. USD', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Capital Flow in bn. USD', fontsize=22)
    plt.xlabel('Years', fontsize=22)
    plt.show()


# average_fund_flow_per_year()


def count_morningstar_cate():
    df = pd.read_csv('data/Morningstar_data_version_5.0.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    df_ia = df.filter(regex='Morningstar Category_')
    cols = list(df_ia.filter(regex='Morningstar Category').columns[:])
    counts = []
    for k in cols:
        value = int(df[k].sum() / 264)
        counts.append(value)

    cols = [x.split('_')[1] for x in cols]

    plt.figure(figsize=(20, 18))
    plt.grid(alpha=0.6)
    plt.bar(cols, counts, color='#3072a1', alpha=0.99)
    plt.title('Number of Funds per Morningstar Category', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Observations', fontsize=22)
    plt.xlabel('Morningstar Category', fontsize=22)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.6)
    plt.show()


# count_morningstar_cate()


def expense_ratio_to_year():
    df = pd.read_csv('data/Morningstar_data_version_5.0_lagged_noDummies.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    df = df[(df["fund_flow"] > 0) | (df["fund_flow"] < 0)]
    df = df[(df["monthly_exp"] > 0)]
    df['monthly_exp'] = df["monthly_exp"] * 12

    # print(df.columns[:])

    y = df['monthly_exp'][:]
    x = df['year'][:]

    sns.set(font_scale=1.3)
    plot = sns.jointplot(data=df, x=x, y=y, kind='reg', line_kws={"color": "#ffa412"})
    plt.title('Distribution of Yearly Expense Ratio to Year', y=1.2, x=-3, fontsize=22)
    plot.ax_joint.set_xlabel('Year', fontsize=18)
    plot.ax_joint.set_ylabel('Yearly Expense Ratio in Percent', fontsize=18)
    plt.tight_layout()
    plt.show()


def fund_flow_to_expense():
    df = pd.read_csv('data/Morningstar_data_version_5.0_lagged_noDummies.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df = df.fillna(0.0)
    df = df[(df["fund_flow"] > 0) | (df["fund_flow"] < 0)]
    df = df[(df["fund_flow"] > -8_000_0) | (df["fund_flow"] < 8_000_0)]
    df = df[(df["fund_flow"] < -10000)]
    df = df[(df["fund_flow"] > -1000000)]
    df['monthly_exp'] = df["monthly_exp"] * 12
    df = df[(df["monthly_exp"] < 4)]
    df = df[(df["monthly_exp"] > 0)]

    df = df.sample(n=100_000)

    y = df['fund_flow'][:]
    x = df['monthly_exp'][:]

    sns.set(rc={'figure.figsize': (25, 15)})
    sns.set(font_scale=1.3)

    plot = sns.jointplot(data=df, x=x, y=y, kind="reg", line_kws={"color": "#ffa412"})

    plt.title('Expense Ratio to Fund Flow', y=1.2, x=-3, fontsize=22)
    plot.ax_joint.set_xlabel('Expense Ratio in Percent', fontsize=18)
    plot.ax_joint.set_ylabel('Monthly Fund Flow', fontsize=18)
    # plt.savefig('xx.png', dpi=500)
    plt.show()


# fund_flow_to_expense()


def fund_flow_to_mgmt_expense():
    df = pd.read_csv('data/Morningstar_data_version_5.0_lagged_noDummies.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df = df.fillna(0.0)
    df = df[(df["fund_flow"] > 0) | (df["fund_flow"] < 0)]
    df = df[(df["fund_flow"] > -8_000_0) | (df["fund_flow"] < 8_000_0)]
    df = df[(df["fund_flow"] < 10_000_000)]
    df = df[(df["fund_flow"] > 10000)]

    df = df.sample(n=100_000)

    y = df['fund_flow']
    x = df['Management \nFee']

    sns.set(rc={'figure.figsize': (25, 15)})
    sns.set(font_scale=1.3)

    plot = sns.jointplot(data=df, x=x, y=y, kind="reg", line_kws={"color": "#ffa412"})

    plt.title('Management Fee to Fund Flow', y=1.2, x=-3, fontsize=22)
    plot.ax_joint.set_xlabel('Management Fee in Percent', fontsize=18)
    plot.ax_joint.set_ylabel('Monthly Fund Flow', fontsize=18)
    # plt.savefig('xx.png', dpi=500)
    plt.show()


def feature_importance(x, y, model):

    df = pd.DataFrame(list(zip(x, y)), columns=['Name', 'Value'])
    df = df.sort_values('Value', ascending=True)
    df = df.reset_index()
    df.drop(['index'], axis=1, inplace=True)
    df = df[-20:]

    plt.figure(figsize=(20, 20))
    plt.grid(alpha=0.6)
    plt.barh(df['Name'], df['Value'], color='#3072a1', alpha=0.99)
    plt.title(f'Feature Importance of {model} Model\n', fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('\nFeatures', fontsize=24)
    plt.xlabel('\nImportance', fontsize=24)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.6)
    plt.tight_layout()
    plt.show()


def confusion_matrix(matrix, model):
    ax = sns.heatmap(matrix, annot=True, cmap='Blues')

    ax.set_title(f'Confusion Matrix for {model} Model\n', fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_xlabel('\nPredicted Values', fontsize=24)
    ax.set_ylabel('\nActual Values ', fontsize=24)

    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    plt.show()


def loss_visualizer(train_loss, val_loss, epochs):
    plt.plot(range(1, epochs+1), train_loss, label='Training loss')
    plt.plot(range(1, epochs+1), val_loss, label='validation loss')
    plt.title('Training and Validation loss\n', fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('\nEpochs', fontsize=24)
    plt.ylabel('\nLoss', fontsize=24)
    plt.legend()
    plt.tight_layout()
    plt.show()


def accuracy_visualizer(train_acc, val_acc, epochs):
    plt.plot(range(1, epochs+1), train_acc, label='Training accuracy')
    plt.plot(range(1, epochs+1), val_acc, label='validation accuracy')
    plt.title('Training and Validation accuracy\n', fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('\nEpochs', fontsize=24)
    plt.ylabel('\nAccuracy', fontsize=24)
    plt.legend()
    plt.tight_layout()
    plt.show()


def regression_baseline():
    df = dsc.ml_algo_selection('regression')
    average_ff = round(df['fund_flow'].mean(), 0)

    print(average_ff)