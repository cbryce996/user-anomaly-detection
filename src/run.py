import pandas as pd
import numpy as np
from pathlib import Path
from plot import plot_feature_scores, plot_heatmap, plot_bar, plot_roc_curve
from model import build_pipelines, select_k_best, fit_model, validate_model
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Select best features
best = select_k_best(ds_train_X, ds_train_y, 15)

# Remove reduntant
ds_test_X = ds_test_X.iloc[:,best['columns']]
ds_train_X = ds_train_X.iloc[:,best['columns']]

# Define algorithms
algos = {
    'Logistic Regression': LogisticRegression(n_jobs=-1, max_iter=5000),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'K-Nearest Neighbor': KNeighborsClassifier(n_jobs=-1),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB()
}

# Create models
models = build_pipelines(algos)

# Fit and score models
scores = pd.DataFrame(
    index=[],
    columns=[
        'Loss',
        'Accuracy',
        'Precision',
        'Recall',
        'F1',
        'False Positive',
        'True Positive'
    ]
)
for algo in models:
    model = models[algo]
    fit = fit_model(model, ds_train_X, ds_train_y)
    val = validate_model(fit, ds_test_X, ds_test_y)
    row = [
        val['loss'],
        val['accuracy'],
        val['precision'],
        val['recall'],
        val['f1'],
        val['roc']['fp'],
        val['roc']['tp']
    ]
    scores.loc[algo] = row
plot_bar(scores)