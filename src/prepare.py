import logistic_regression as lr
import support_vector_machine as svm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, normalize

# Import data
test = pd.read_csv('../assets/data/UNSW-NB15/UNSW_NB15_testing-set.csv')
train = pd.read_csv('../assets/data/UNSW-NB15/UNSW_NB15_training-set.csv')

# Remove missing/unwanted columns
test = test.loc[:, test.columns!='attack_cat']
train = train.loc[:, train.columns!='attack_cat']

# Encode categorical data
objs = test.select_dtypes(include=['object']).copy()

for column in objs:
    test[column] = test[column].astype('category')
    test[column] = test[column].cat.codes

objs = train.select_dtypes(include=['object']).copy()

for column in objs:
    train[column] = train[column].astype('category')
    train[column] = train[column].cat.codes

# Scale
train_X = normalize(train.iloc[:,:-1])
test_X = normalize(test.iloc[:,:-1])

lr.run_logistic_regression(train_X, train.iloc[:,-1], test_X, test.iloc[:,-1])