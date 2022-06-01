import pandas as pd  # The pandas library is used for working with and manipulation of data files, in this case .csvs
import warnings  # the warnings library allows for the manipulation (or exclusion) of python warnings
pd.set_option('display.max_columns', 500)  # show more columns in the terminal output
pd.set_option('display.max_rows', 500)  # show more rows in the terminal output
pd.options.mode.chained_assignment = None  # default='warn'  # Turns of the 'setting with copy warning' -> Cleaner term.
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)  # Turns off performance warnings

"""

DESCRIPTION OF MODULE:
------------------------------------------------------------------------------------------------------------------------
This module is used for the creation of a clean and workable dataset. The files downloaded from Morningstar are 
combined previously. 

Other usages of this module:
    - Drop the variable with not enough data points in them
    - Drop observations if not enough fund flow data points are included 
    - Convert string variables to dummies 
    - Remove observations if funds are younger than 3 years
    - Conversion of yearly data points to monthly in order to match them with other variables 
    - Conversion of the dataset to panel data 
    - Algorithm selection function that retrieves the right dataset depending in the algorithm used
        --> Classification versus regression 

"""


def data_cleaning():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This function drops various columns based on the lack of data points within them.

    """

    df = pd.read_csv('data/Morningstar_data_version_1.1.csv')  # reads the exported file from Morningstar

    # Alpha is dropped because it is calculated through regression at a later stage. The preexisting alpha from
    # Morningstar as well as the beta should not be used.
    # Public, Number of Shareholders, Net Expense Ratio, TER, and IPO NAV all had very few data points available.
    drops = ['Alpha 1 Yr (Gross Return)(Qtr-End)', 'Public', 'Number of Shareholders', 'Net Expense \nRatio',
             'Total Expense Ratio', 'IPO NAV']

    for drop in drops:
        try:
            df.drop([drop], axis=1, inplace=True)
        except:
            print(f'Error dropping: {drop}')

    regex_drops = ['MER', 'Unnamed', 'Morningstar Analyst', 'P/E Ratio (TTM)', 'beta']

    for regex_drop in regex_drops:
        try:
            df.drop(list(df.filter(regex=regex_drop)), axis=1, inplace=True)
        except:
            print(f'Error dropping: {regex_drop}')
    df = df.replace('’', '', regex=True)

    return df


def drop_if_not_enough_ff_data():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Because machine learning algorithms need as clean data as possible, those fund observations that only inherit very
    little fund flow data points are dropped. In this case, too little data points are defined as having less than 36.

    """
    df = data_cleaning()  # retrieve the dataset from the function above.
    df.insert(4, 'ff_data_points', '')  # insert a column for the number of data points a fund has

    # These two lists are used for a nested loop that iterates over the Fund Flow columns, as there are 12 * 21 fund
    # flow columns in the dataset.
    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))

    for year in list_years:
        for month in list_months:
            col = f'Estimated Share Class Net Flow (Monthly) \n{year}-{month} \nBase \nCurrency'
            df[col] = df[col].fillna(0.0)  # if the datapoint is empty, fill it with 0, because can't calc with nans.
            df[col] = df[col].astype(float)  # convert all the values in a column to floats

    # Count the number of data points for a fund and drop all observations with less than 36 data points.
    for i in range(len(df.index[:])):
        ff_data = []
        for year in list_years:
            for month in list_months:
                ff_column = f'Estimated Share Class Net Flow (Monthly) \n{year}-{month} \nBase \nCurrency'
                value = df[ff_column][i]
                ff_data.append(value)
        sum_ff_points = int(sum(x > 0 or x < 0 for x in ff_data))  # list comprehension that sums the occurrences.
        df['ff_data_points'][i] = sum_ff_points
        print(f'Setting ff_data point for: {i} --> done!')  # verbose output to check on the progress.

    df = df.drop(df[df.ff_data_points < 36].index)

    print(len(df.index))  # For checking whether everything worked as anticipated.

    return df


def remove_younger_than_3_years():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    As described in the paper, all funds that are instantiated less than 36 months ago have to be dropped in order to
    eliminate biases.

    """
    df = drop_if_not_enough_ff_data()  # retrieve dataset from previous function
    # Only use the observations that have an inception date and drop those who do not have one.
    df = df[df['Inception \nDate'].notna()]
    df['Inception \nDate'] = pd.to_datetime(df['Inception \nDate'], format='%Y-%m-%d')  # convert to datetime object

    # Exactly 3 months into the past from the point of downloading the dataset.
    df = df[~(df['Inception \nDate'] > '2019-04-01')]

    return df


def remove_many_nans():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    As other features also have too few data points, it was decided to drop all of those features too instead of
    trying fill them in with mean / medians or zeros. Instead, dropping these features of more correct.

    """
    df = remove_younger_than_3_years()  # Again, taking the dataset from the previous function

    # The 'Unnamed' columns in only an index column created automatically by pandas. The row below will be featured
    # in various functions throughout this project because pandas keeps converting the index to another column.
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    # List of columns that are to be dropped. Among them are another column displaying the beta over time which, again,
    # will be calculated by the author himself. Also, two columns from fund flow data points occurring in 2022
    # are included in the Morningstar export. These are not used and therefore can be dropped.
    col_drops = ['Time Horizon', '# of \nStock \nHoldings (Short)',
                 'Average Market Cap \n(mil) (Short) \nPortfolio \nCurrency', 'ff_data_points', 'Performance Fee',
                 'Beta \n2000-01-01 \nto 2022-03-31 \nBase \nCurrency',
                 'Estimated Share Class Net Flow (Monthly) \n2022-01 \nBase \nCurrency',
                 'Estimated Share Class Net Flow (Monthly) \n2022-02 \nBase \nCurrency']
    for col_drop in col_drops:
        df.drop([col_drop], axis=1, inplace=True)

    # Some features, however, are vital for this paper and hence the missing data points to not grant dropping them
    # completely. Instead, the observations with the missing data points are eliminated.
    row_drops = ['Net Assets \n- Average', 'Manager \nTenure \n(Longest)', 'Percent of Female Executives', 'Firm City',
                 'P/E Ratio (TTM) (Long)', 'Investment Area', 'Management \nFee']
    for row_drop in row_drops:
        df = df[df[row_drop].notna()]

    return df


def convert_return_data():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    This function will create the panel data structure for the monthly returns variable used for the machine learning
    algorithms. There is a preexisting function from the pandas library that is called 'melt'.

    """
    df = remove_many_nans()

    list_cols = ['Name']
    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.
    for year in list_years:
        for month in list_months:
            list_cols.append(f'Monthly Gross Return \n{year}-{month} \nBase \nCurrency')

    df_returns = pd.melt(frame=df[list_cols], id_vars=['Name'], var_name="year-month", value_name='monthly_return')

    year = df_returns["year-month"].str[22:26].astype(int)
    month = df_returns["year-month"].str[27:29].astype(int)

    df_returns.insert(2, 'year', year)
    df_returns.insert(2, 'month', month)

    df_returns.drop(['year-month'], axis=1, inplace=True)

    return df_returns


def convert_to_panel_data():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Here, the remaining columns are converted into panel data and the return panel dataset is inserted.

    """
    df_returns = convert_return_data()
    df = remove_many_nans()

    list_cols = []
    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    list_years = list(range(2000, 2022))  # --> creates a list of values from 2000 to 2021.
    for year in list_years:
        for month in list_months:
            list_cols.append(f'Estimated Share Class Net Flow (Monthly) \n{year}-{month} \nBase \nCurrency')

    drops = df.loc[:, ~df.columns.isin(list_cols)]
    drops = list(drops.columns[:])

    # The pandas built-in function that creates panel datasets.
    df_panel = pd.melt(frame=df, id_vars=drops, var_name="year-month", value_name='fund_flow')

    # Extracting the year and month from the dataframe.
    year = df_panel["year-month"].str[42:46].astype(int)
    month = df_panel["year-month"].str[47:50].astype(int)

    # Inserting the year and month into the dataframe.
    df_panel.insert(2, 'year', year)
    df_panel.insert(2, 'month', month)

    # Dropping the Monthly Gross Return column as it is now converted into a panel dataset.
    df_panel.drop(list(df.filter(regex='Monthly Gross Return')), axis=1, inplace=True)

    # Inserting the correct, converted column into the panel dataset.
    monthly_return = df_returns.pop('monthly_return')
    df_panel.insert(40, 'monthly_return', monthly_return)

    # Saving the dataset for analysis.
    df_panel.to_csv('data/Morningstar_data_version_2.1.csv')

    return df_panel


def dummy_variables():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Convert the string variables 'Investment Area', 'Morningstar Category', 'Firm City' into dummy variables.

    """
    df = convert_to_panel_data()  # --> This is the panel dataset from the function above.
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)  # Again, dropping the unnamed, empty column.
    df.drop(['year-month'], axis=1, inplace=True)  # Drop the variable by which the data was converted into a panel.

    dummy_list = ['Investment Area', 'Morningstar Category', 'Firm City']  # List of dummy variables to be converted.

    df = pd.get_dummies(df, columns=dummy_list, drop_first=False)  # Convert the dummy variables.

    return df  # Return the dataframe.


def convert_annual_expenses():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Convert the annual expense ratios into monthly data points.

    """
    df = remove_many_nans()
    df = df.reset_index()  # Resetting the index.

    # Months in this dataset are not integers but string and hence the list was created from sting values.
    list_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    # Years in this dataset are integers and hence we can use the list(range(x,y) combination to create a list of years.
    list_years = list(range(2000, 2022))
    list_cols = ['Name']  # The reference column that holds the fund names
    for year in list_years:
        col = f'Annual Report Net Expense Ratio \nYear{year}'
        list_cols.append(col)  # Create a list of all the columns that hold annual expense data

    df_annum_exp = df[list_cols]
    df_annum_exp = df_annum_exp.fillna(0.0)  # when the fund was non-existing, the annual expense ratio was set to 0
    for year in list_years:
        for month in list_months:
            df_annum_exp.insert(1, f"monthly_exp_ratio_{year}_{month}", "")  # create a column for each month & year

    for i in range(len(df_annum_exp.index)):
        for year in list_years:
            annual = df_annum_exp[f'Annual Report Net Expense Ratio \nYear{year}'][i]
            monthly = round(annual / 12, 6)  # round the value to six digits after the dot
        for year in list_years:
            for month in list_months:
                df_annum_exp[f"monthly_exp_ratio_{year}_{month}"][i] = monthly

    df_annum_exp.drop(list(df_annum_exp.filter(regex='Annual Report Net Expense Ratio')), axis=1, inplace=True)

    list_cols = ['Name']
    for year in list_years:
        for month in list_months:
            list_cols.append(f'monthly_exp_ratio_{year}_{month}')

    # Convert the dataframe into a panel dataset.
    df_exp = pd.melt(frame=df_annum_exp[list_cols], id_vars=['Name'], var_name="year-month", value_name='monthly_exp')

    # Again, drop the year-month column only used for the conversion.
    df_exp.drop(['year-month'], axis=1, inplace=True)

    return df_exp


def concat_maindf_and_expdf():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Concat the previously create monthly expense panel dataset with the overall dataset. After this step, the data
    cleaning process is complete and one can begin with the feature engineering part. Feature engineering is the
    practice of creating meaningful columns for the machine learning algorithms.

    """
    df = dummy_variables()  # get the dataset will all variables converted into dummies.
    df_exp = convert_annual_expenses()

    monthly_exp = df_exp.pop('monthly_exp')
    df.insert(9, 'monthly_exp', monthly_exp)

    df.drop(list(df.filter(regex='Annual Report Net Expense Ratio')), axis=1, inplace=True)  # Not needed anymore.

    return df


def ml_algo_selection(ml_type):
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    When calling this function, the user can choose the type of algorithm that is being used. The choices are:
        - regression, classification, and extended classification.
            - regression is aiming at predicting the actual value of the fund flow
            - classification is the binary classification of whether a fund flow is positive (==1) or not (==0)
            - extended classification is the classification of 21 different classes based on their fund flow
              magnitude and whether these flows are positive or negative

    The final, usable dataset for the corresponding algorithm is then returned to the algorithm functions.

    """
    data = pd.read_csv('data/Morningstar_data_version_5.0_lagged.csv')  # Retrieve the dataset
    data.drop(list(data.filter(regex='Unnamed')), axis=1, inplace=True)  # Drop unnamed columns, dont have data

    # Removing the columns that were not convertable into dummy variables. Also dropping inception data here because
    # it is already converted into numeric categories as month and year.
    data.drop(['Management Company', 'Name', 'Inception \nDate'], axis=1, inplace=True)  # String features removed

    # Renaming all the column names since they are not lagged.
    data = data.rename(columns={'Manager \nTenure \n(Average)': 'Avg. Manager Tenure',
                                'Manager \nTenure \n(Longest)': 'Max. Manager Tenure',
                                'Net Assets \n- Average': 'Avg. Net Assets',
                                'Average Market Cap (mil) (Long) \nPortfolio \nCurrency': 'Avg. Market Cap',
                                'Management \nFee': 'Management Fee'})
    for i, k in enumerate(list(data.columns[:])):
        data = data.rename(columns={list(data.columns[:])[i]: f'{list(data.columns[:])[i]} lagged'})

    # The predicted variable of course is not lagged and hence it is re-renamed to just 'fund_flow'.
    data = data.rename(columns={'fund_flow lagged': 'fund_flow'})

    if ml_type == 'regression':
        return data  # this dataset does not have to be modified any further

    elif ml_type == 'classifier':

        def ff_positive(x):
            if x >= 0.0:
                return 1
            elif x < 0.0:
                return 0

        data['fund_flow'] = data['fund_flow'].apply(ff_positive)

        return data  # returning the dataset that inherits the binary values for fund flows

    elif ml_type == 'extended_classifier':

        # Classification of all fund flow values leads to very poor results as the values are diverging too far apart
        # from each other. Hence, the fund flows are cut of at the tails in this case chosen to be -10 and 10 million.
        data = data[(data["fund_flow"] < 10_000_000)]
        data = data[(data["fund_flow"] > -10_000_000)]
        data = data[(data["fund_flow"] != 0)]

        min_ff = data['fund_flow'].min()  # minimum value of the fund flow included
        max_ff = data['fund_flow'].max()  # maximum value of the fund flow included

        # Function for the advanced classification into positive and negative classes as well as their magnitude.
        def categorize_ff(val):
            if val >= max_ff * 0.60:
                return 10
            elif val >= max_ff * 0.40:
                return 9
            elif val >= max_ff * 0.35:
                return 8
            elif val >= max_ff * 0.30:
                return 7
            elif val >= max_ff * 0.25:
                return 6
            elif val >= max_ff * 0.20:
                return 5
            elif val >= max_ff * 0.15:
                return 4
            elif val >= max_ff * 0.10:
                return 3
            elif val >= max_ff * 0.05:
                return 2
            elif val > max_ff * 0.00:
                return 1
            elif val == max_ff * 0.00:
                return 0
            elif val >= min_ff * 0.05:
                return -1
            elif val >= min_ff * 0.10:
                return -2
            elif val >= min_ff * 0.15:
                return -3
            elif val >= min_ff * 0.20:
                return -4
            elif val >= min_ff * 0.30:
                return -5
            elif val >= min_ff * 0.40:
                return -6
            elif val >= min_ff * 0.50:
                return -7
            elif val >= min_ff * 0.60:
                return -8
            elif val >= min_ff * 0.80:
                return -9
            elif val < min_ff * 0.80:
                return -10

        data['fund_flow'] = data['fund_flow'].apply(categorize_ff)

        return data  # returning the dataset that inherits the advanced classes for fund flows


def reconvert_dummies():
    """

    DESCRIPTION:
    --------------------------------------------------------------------------------------------------------------------
    Reconverting the dummy variables to numerically categorized ones for visualization purposes. As it is too
    difficult to visualize 200 variables, all the dummies are reconverted into classes and then mapped with their
    respective value. Makes for better and more interpretable visualization.

    """
    df = pd.read_csv('data/Morningstar_data_version_5.0_lagged.csv')  # retrieving the dataset
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)

    # Converting the Investment Area dummy variable.
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

    # Converting the Morningstar category dummy variable.
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

    # Converting the Firm City dummy variable.
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

    # Creating dataframes of the converted classes and inserting them into the primary dataset.
    df_ia = convert_Investment_Area()
    df_mc = convert_Morningstar_Category()
    df_fc = convert_Firm_City()

    Investment_Area = df_ia.pop('Investment Area')
    df.insert(9, 'Investment Area', Investment_Area)

    Morningstar_Category = df_mc.pop('Morningstar Category')
    df.insert(9, 'Morningstar Category', Morningstar_Category)

    Firm_City = df_fc.pop('Firm City')
    df.insert(9, 'Firm City', Firm_City)

    # Dropping all the dummy variable columns.
    df.drop(list(df.filter(regex='Investment Area_')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Morningstar Category_')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Firm City_')), axis=1, inplace=True)

    return df