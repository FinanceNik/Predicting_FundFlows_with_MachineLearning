import pandas as pd
import datetime as dt
import DataSet_Cleaner as dsc
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def get_fama_french_data():
    import pandas_datareader.data as reader

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

    start = dt.date(2000, 1, 1)
    end = dt.date(2021, 12, 31)

    df_fama = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0]
    df = transform_return_df()

    col_pops = ['Mkt-RF', 'SMB', 'HML', 'RF']
    for col in col_pops:
        col_popped = df_fama.pop(col)
        df.insert(0, col, col_popped.values)

    df.to_csv('Alpha_Calculation_Dataset.csv')

    return df


def calculate_alpha():
    df = pd.read_csv('Alpha_Calculation_Dataset.csv')
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
        print(f" stage_excess --> {count_excess}")

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

            df[f'{fund}_alpha'][i_row] = alpha
            df[f'{fund}_beta'][i_row] = beta

            if df[f"{fund}"][i_row] == 0:
                df[f'{fund}_alpha'] = 0.0
                df[f'{fund}_beta'] = 0.0

        count_regression = count_regression+1
        print(f" stage_regression --> {count_regression}")

    include_cols = [i for i in list(df) if 'alpha' in i
                    or 'beta' in i
                    or 'Mkt' in i
                    or 'SMB' in i
                    or 'HML' in i
                    or 'RF' in i]

    df_final = df[include_cols]

    df_final.to_csv('Alpha_AND_Beta_Calculation_Finalized.csv')


def create_alpha_df():
    df = pd.read_csv('Alpha_AND_Beta_Calculation_Finalized.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='beta')), axis=1, inplace=True)

    df = df.fillna(0.0)
    df = df.T

    df.insert(0, 'Name', '')
    df['Name'] = df.index
    df = df.reset_index()

    df.to_csv('alpha.csv')

    return df


def create_beta_df():
    df = pd.read_csv('Alpha_AND_Beta_Calculation_Finalized.csv')
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='alpha')), axis=1, inplace=True)

    df = df.fillna(0.0)
    df = df.T

    df.insert(0, 'Name', '')
    df['Name'] = df.index
    df = df.reset_index()

    df.to_csv('beta.csv')

    return df


def transform_alpha_AND_beta():
    df = pd.read_csv('data/Morningstar_data_version_3.0.csv')
    df_alpha = pd.read_csv('alpha.csv')
    df_alpha.drop(list(df_alpha.filter(regex='Unnamed')), axis=1, inplace=True)
    df_alpha = df_alpha.iloc[::-1]
    df_alpha = df_alpha.reset_index()
    df_beta = pd.read_csv('beta.csv')
    df_beta.drop(list(df_beta.filter(regex='Unnamed')), axis=1, inplace=True)
    df_beta = df_beta.iloc[::-1]
    df_beta = df_beta.reset_index()

    list_cols_alpha = list(df_alpha.columns[1:])

    df_alpha_final = pd.melt(frame=df_alpha[list_cols_alpha], id_vars=['Name'],
                             var_name="remove", value_name='monthly_alpha')

    list_cols_beta = list(df_alpha.columns[1:])

    df_beta_final = pd.melt(frame=df_alpha[list_cols_beta], id_vars=['Name'],
                             var_name="remove", value_name='monthly_beta')

    monthly_alpha = df_alpha_final.pop('monthly_alpha')
    df.insert(1, 'monthly_alpha', monthly_alpha)

    monthly_beta = df_beta_final.pop('monthly_beta')
    df.insert(1, 'monthly_beta', monthly_beta)

    df.to_csv('data/Morningstar_data_version_4.0.csv')






