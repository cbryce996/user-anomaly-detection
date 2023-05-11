import feature_selection as fs
import pandas as pd
from pathlib import Path

# Import data
ds = pd.read_csv('../assets/data/UNSW-NB15/UNSW_NB15_training-set.csv')

# Split data X
ds_X = ds.iloc[:,:-1]

# Split data y
ds_y = ds.iloc[:,-1]

# Drop unwanted columns
ds_X.drop(['attack_cat'], axis=1, inplace=True)

# Clean data
ds_X.fillna(0, inplace=True)

# Encode categorical data
objs = ds_X.select_dtypes(include=['object']).copy()
for column in objs:
    ds_X[column] = ds_X[column].astype('category')
    ds_X[column] = ds_X[column].cat.codes

# Perform feature selection
fs.run(ds_X, ds_y)