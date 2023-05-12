import pandas as pd
from pathlib import Path
from plot import plot_feature_scores
from model import build_pipelines, select_k_best
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Import data
ds_train = pd.read_csv('../assets/data/UNSW-NB15/UNSW_NB15_training-set.csv')
ds_test = pd.read_csv('../assets/data/UNSW-NB15/UNSW_NB15_testing-set.csv')

# Split data X
ds_train_X = ds_train.iloc[:,:-1]
ds_test_X = ds_test.iloc[:,:-1]

# Split data y
ds_train_y = ds_train.iloc[:,-1]
ds_test_y = ds_test.iloc[:,-1]

# Drop unwanted columns
for ds in (ds_train_X, ds_test_X):
    ds.drop(['attack_cat'], axis=1, inplace=True)
    ds.drop(['id'], axis=1, inplace=True)

# Clean data
for ds in (ds_train_X, ds_test_X):
    ds.fillna(0, inplace=True)

# Encode categorical data
for ds in (ds_train_X, ds_test_X):
    objs = ds.select_dtypes(include=['object']).copy()
    for column in objs:
        ds[column] = ds[column].astype('category')
        ds[column] = ds[column].cat.codes

# Define algorithms
algos = {
    'Logistic Regression': LogisticRegression(n_jobs=-1, max_iter=5000),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'K-Nearest Neighbor': KNeighborsClassifier(n_jobs=-1)
}

# Create models
models = build_pipelines(algos)

# Select best features
best = select_k_best(ds_train_X, ds_train_y, 10)

# Plot feature scores
scores = pd.DataFrame(
    {
        'features': ds_test_X.columns,
        'scores': best['scores']
    }
)
plot_feature_scores(scores.sort_values('scores', ascending=False))

# Remove reduntant
ds_test_X = ds_test_X.iloc[:,best['columns']]
ds_train_X = ds_train_X.iloc[:,best['columns']]