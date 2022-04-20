import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('data/Morningstar_data_version_1.4_filtered_numOnly.csv')
# For some reason beyond me, pandas is trying to re-set a new index column every time that df is instantiated.
# Because these are always named 'Unnamed: {}', one can easily filter them with a regular expression.
df = df.drop(list(df.filter(regex='Unnamed')), axis=1)

print(df.describe().transpose().round(2))