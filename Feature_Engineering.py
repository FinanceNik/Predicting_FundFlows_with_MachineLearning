import pandas as pd
import datetime as dt
import DataSet_Cleaner as dsc
import statsmodels.api as sm  # the stats-model library is used for the regression functions.
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

"""

DESCRIPTION OF MODULE:
------------------------------------------------------------------------------------------------------------------------

This module is concerned with creating features (columns) that aid the machine learning algorithms in their predictive
ability. Mainly, this module inserts the Fama French Factors into the panel dataset as well as calculates alpha and 
beta for all of the funds included. 

- Fetch the Fama French 3-Factor model factors and create a dataframe object out of them
- Calculate rolling alpha and beta for the funds 
- Lag all of the predicting variables 

"""


def get_fama_french_data():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Retrieving the Fama French factors from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    by using pandas datareader framework. In the end the dataset is returned as a dataframe object.

    """
    import pandas_datareader.data as reader  # for reading html and web-based data.

    def transform_return_df():
        df_transposed = dsc.remove_many_nans()
        df_transposed.drop(list(df_transposed.filter(regex='Unnamed')), axis=1, inplace=True)

        list_cols = ['Name']
        list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.
        for year in list_years:
            for month in list_months:
                list_cols.append(f'Monthly Gross Return \n{year}-{month} \nBase \nCurrency')

        df_transposed = df_transposed[list_cols]
        df_transposed = df_transposed.fillna(0.0)
        df_transposed = df_transposed.set_index('Name').T

        return df_transposed

    # The relevant timeframe used for the research in this paper.
    start = dt.date(2000, 1, 1)
    end = dt.date(2021, 12, 31)

    # Instantiating the reader class so that pandas can access the fama-french factors from the above-mentioned URL
    df_fama = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0]
    df = transform_return_df()

    col_pops = ['Mkt-RF', 'SMB', 'HML', 'RF']  # the factors included in the 3-factor model
    for col in col_pops:
        col_popped = df_fama.pop(col)
        df.insert(0, col, col_popped.values)  # inserting the columns with the fama-french factors into the dataframe

    return df  # returning the dataframe with the fama-french factors


def calculate_alpha_and_beta():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This function is the core of the alpha and beta calculation. Here, alpha and beta are estimated using a rolling
    regression approach. Firstly, calculate the excess return of each fund for each period and then the alpha and beta
    for each period. Only of a fund does not have any excess return data at a given point in time, do not calculate
    alpha and beta.

    """
    df = get_fama_french_data()
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    count_excess = 0
    for fund in df.columns[4:]:
        df[f"{fund}_excess"] = df[fund] - df.RF
        for i in range(len(df.index)):
            if df[f"{fund}"][i] == 0:
                try:
                    df[f"{fund}_excess"][i] = 0.0
                except:
                    pass
        df.drop([fund], axis=1, inplace=True)
        count_excess = count_excess + 1

    count_regression = 0
    for fund in df.columns[4:]:
        df.insert(1, f'{fund}_alpha', '')
        df.insert(1, f'{fund}_beta', '')

        for i_row in range(1, len(df.index)):
            y = df[f"{fund}"][:i_row]
            X = df[['Mkt-RF', 'SMB', 'HML', 'RF']][:i_row]
            X_sm = sm.add_constant(X)
            model = sm.OLS(y, X_sm)
            results = model.fit()
            coeff = results.params
            alpha = round(coeff[0], 8)  # Alpha for the whole period for fund_i
            beta = round(coeff[1], 8)  # Beta for the whole period for fund_i

            df[f'{fund}_alpha'][i_row], df[f'{fund}_beta'][i_row] = alpha, beta

            if df[f"{fund}"][i_row] == 0:
                df[f'{fund}_alpha'] = 0.0
                df[f'{fund}_beta'] = 0.0

        count_regression = count_regression+1

    include_cols = [i for i in list(df) if 'alpha' in i
                    or 'beta' in i
                    or 'Mkt' in i
                    or 'SMB' in i
                    or 'HML' in i
                    or 'RF' in i]

    df_final = df[include_cols]

    return df_final  # returning the finalized dataframe object


def create_alpha_and_beta_df():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Creating two separate datasets for alpha and beta values in order to transform and insert them into the main
    data-frame in another function later.

    """
    df_alpha = calculate_alpha_and_beta()
    df_alpha.drop(list(df_alpha.filter(regex='Unnamed')), axis=1, inplace=True)
    df_alpha.drop(list(df_alpha.filter(regex='beta')), axis=1, inplace=True)

    df_alpha = df_alpha.fillna(0.0).T  # transpose the dataframe after filling the NaNs

    df_alpha.insert(0, 'Name', '')
    df_alpha['Name'] = df_alpha.index
    df_alpha = df_alpha.reset_index()  # resetting the index

    df_beta = calculate_alpha_and_beta()
    df_beta.drop(list(df_beta.filter(regex='Unnamed')), axis=1, inplace=True)
    df_beta.drop(list(df_beta.filter(regex='alpha')), axis=1, inplace=True)

    df_beta = df_beta.fillna(0.0).T  # transpose the dataframe after filling the NaNs

    df_beta.insert(0, 'Name', '')
    df_beta['Name'] = df_beta.index
    df_beta = df_beta.reset_index()  # resetting the index

    return df_alpha, df_beta  # returning both dataframe objects


def transform_alpha_AND_beta():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    The alpha and beta dataframes are converted into panel data and inserted into the main dataset.

    """
    df = dsc.concat_maindf_and_expdf()  # the final panel dataset created in the DataSet_Cleaner module
    df_alpha = create_alpha_and_beta_df()[0]
    df_alpha.drop(list(df_alpha.filter(regex='Unnamed')), axis=1, inplace=True)  # dropping unnamed, empty column
    df_alpha = df_alpha.iloc[::-1]  # reverse the dataset to fit the primary dataset
    df_alpha = df_alpha.reset_index()
    df_beta = create_alpha_and_beta_df()[1]
    df_beta.drop(list(df_beta.filter(regex='Unnamed')), axis=1, inplace=True)  # dropping unnamed, empty column
    df_beta = df_beta.iloc[::-1]  # reverse the dataset to fit the primary dataset
    df_beta = df_beta.reset_index()

    list_cols_alpha = list(df_alpha.columns[1:])

    # Converting the dataframe into panel data.
    df_alpha_final = pd.melt(frame=df_alpha[list_cols_alpha], id_vars=['Name'],
                             var_name="remove", value_name='monthly_alpha')

    list_cols_beta = list(df_beta.columns[1:])

    # Converting the dataframe into panel data.
    df_beta_final = pd.melt(frame=df_beta[list_cols_beta], id_vars=['Name'],
                             var_name="remove", value_name='monthly_beta')

    monthly_alpha = df_alpha_final.pop('monthly_alpha')
    df.insert(1, 'monthly_alpha', monthly_alpha)  # inserting the cleaned column

    monthly_beta = df_beta_final.pop('monthly_beta')
    df.insert(1, 'monthly_beta', monthly_beta)  # inserting the cleaned column

    return df  # returning the final dataframe object


def insert_factors():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Converting the fama french factors into panel data and inserting them into the main dataset.

    """
    # Insert the Fama-French 3-Factor Model Factors into the dataset.
    df_factors = calculate_alpha_and_beta()
    df_factors.drop(list(df_factors.filter(regex='Unnamed')), axis=1, inplace=True)

    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.

    df = dsc.remove_many_nans()
    df.drop(list(df_factors.filter(regex='Unnamed')), axis=1, inplace=True)

    df_factors = df_factors[['Mkt-RF', 'SMB', 'HML', 'RF']]

    df_factors.insert(1, 'Date', '')

    dates = []
    for year in list_years:
        for month in list_months:
            dates.append(f'{year}_{month}')

    for i in range(len(df_factors.index)):
        df_factors['Date'][i] = dates[i]

    df = df[['Name']]

    for year in reversed(list_years):
        for month in reversed(list_months):
            df.insert(1, f'Mkt-RF_{year}_{month}', '')

    for year in reversed(list_years):
        for month in reversed(list_months):
            df.insert(1, f'SMB_{year}_{month}', '')

    for year in reversed(list_years):
        for month in reversed(list_months):
            df.insert(1, f'HML_{year}_{month}', '')

    for year in reversed(list_years):
        for month in reversed(list_months):
            df.insert(1, f'RF_{year}_{month}', '')

    for year in list_years:
        for month in list_months:
            df[f'Mkt-RF_{year}_{month}'] = float(df_factors.loc[df_factors['Date'] == f'{year}_{month}', 'Mkt-RF'])
            df[f'SMB_{year}_{month}'] = float(df_factors.loc[df_factors['Date'] == f'{year}_{month}', 'SMB'])
            df[f'HML_{year}_{month}'] = float(df_factors.loc[df_factors['Date'] == f'{year}_{month}', 'HML'])
            df[f'RF_{year}_{month}'] = float(df_factors.loc[df_factors['Date'] == f'{year}_{month}', 'RF'])

    # Because the fund name also has to be included as an identifier, it is added as the first element to each list.
    list_cols_MKTRF, list_cols_SMB, list_cols_HML, list_cols_RF = ['Name'], ['Name'], ['Name'], ['Name']

    for year in list_years:
        for month in list_months:
            list_cols_MKTRF.append(f'Mkt-RF_{year}_{month}')
            list_cols_SMB.append(f'SMB_{year}_{month}')
            list_cols_HML.append(f'HML_{year}_{month}')
            list_cols_RF.append(f'RF_{year}_{month}')

    # Converting all the separate dataframes into panel data in order to insert them into the main dataframe.
    df_mktrf = pd.melt(frame=df[list_cols_MKTRF], id_vars=['Name'], var_name="year-month", value_name='mktrf')
    df_mktrf.drop(['year-month'], axis=1, inplace=True)
    
    df_smb = pd.melt(frame=df[list_cols_SMB], id_vars=['Name'], var_name="year-month", value_name='smb')
    df_smb.drop(['year-month'], axis=1, inplace=True)
    
    df_hml = pd.melt(frame=df[list_cols_HML], id_vars=['Name'], var_name="year-month", value_name='hml')
    df_hml.drop(['year-month'], axis=1, inplace=True)
    
    df_rf = pd.melt(frame=df[list_cols_RF], id_vars=['Name'], var_name="year-month", value_name='rf')
    df_rf.drop(['year-month'], axis=1, inplace=True)

    mktrf, smb, hml, rf = df_mktrf.pop('mktrf'), df_smb.pop('smb'), df_hml.pop('hml'), df_rf.pop('rf')

    df_final = transform_alpha_AND_beta()  # the final dataset established before, only without FF factors

    df_final.insert(1, 'mktrf', mktrf)  # insert the market-risk-premium variable into the final dataset
    df_final.insert(1, 'smb', smb)  # insert the small-minus-big factor variable into the final dataset
    df_final.insert(1, 'hml', hml)  # insert the high-minus-low variable into the final dataset
    df_final.insert(1, 'rf', rf)  # insert the risk-free rate  variable into the final dataset

    return df_final


def lag_vars():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    The last step in order to finalize the dataset is to lag all the predicting variables.

    """
    df = insert_factors()
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    # the lag number is 7345 because since the dataframe is already converted to panel data, every fund occurs at
    # exactly the 7345th x interval.
    df['fund_flow'] = df['fund_flow'].shift(7345)

    df.to_csv('data/Morningstar_data_version_5.0_lagged.csv')  # saving the very final dataset used for the ML algos








