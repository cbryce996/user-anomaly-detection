# Internal imports
import plot as plt
import features as ft
import model as md

# External imports
import pandas as pd

# Sklearn algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#######################
# Data Pre-Processing #
#######################

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

#####################
# Feature Selection #
######################

# Select best features
columns, scores = ft.select_k_best(ds_train_X, ds_train_y, 15)

# Plot feature scores
data = pd.DataFrame({'X': scores, 'Y': ds_train_X.columns})
plt.plot_bar(data.sort_values('X', ascending=True), 'feature_scores', 'Feature Scoring - ANOVA', (6,10), 'Score', 'Features')

# Plot full correlation matrix
plt.plot_heatmap(ds_train_X.corr(), 'features_full', 'Correlation Matrix - Pearson', (8, 6))

# Remove reduntant
ds_test_X = ds_test_X.iloc[:,columns]
ds_train_X = ds_train_X.iloc[:,columns]

# Plot selected correlation matrix
plt.plot_heatmap(ds_train_X.corr(), 'features_selected', 'Correlation Matrix - Pearson', (8, 6))

# Scale features
ds_train_X = StandardScaler().fit_transform(ds_train_X)
ds_test_X = StandardScaler().fit_transform(ds_test_X)

######################
# Model Optimization #
######################

# Define algorithms
algos = {
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(n_jobs=-1, max_iter=5000, solver='lbfgs'),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'K-Nearest Neighbor': KNeighborsClassifier(n_jobs=-1)
}

'''
# Plot ROC from defaults
lines = {}
for name, model in algos.items():
    train_roc_curve, test_roc_curve = md.build_roc_curve(ds_train_X, ds_train_y, ds_test_X, ds_test_y, model)
    lines[name] = test_roc_curve

# Plot ROC from defaults
plt.plot_line(lines, 'defaults', 'Reveiver Operating Characteristic (ROC)', (8, 6), 'False Positive Rate', 'True Positive Rate', 'AUC')
'''

# Define params
params = {
    #'Naive Bayes': {}, # No params for Naive Bayes
    'Random Forest': {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1]},
    'Decision Tree Classifier': {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'K-Nearest Neighbor': {'n_neighbors': [80, 120, 160, 200, 240, 280]}
}

# Search parameters
lines = {}
for name, param in params.items():
    model = algos[name]
    train_curves, test_curves = md.param_search(ds_train_X, ds_train_y, ds_test_X, ds_test_y, model, param)
    lines['Training Set'] = train_curves
    lines['Testing Set'] = test_curves
    plt.plot_line(lines, 'params_%s' % (name), 'Parameter Search - %s' % (name), (8, 6), next(iter(param)), 'Accuracy', 'Accuracy')

####################
# Model Evaluation #
####################

# Define optimized algorithms
algos = {
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(max_depth=3),
    'Logistic Regression': LogisticRegression(n_jobs=-1, max_iter=5000, solver='lbfgs', C=0.05),
    'Decision Tree Classifier': DecisionTreeClassifier(max_depth=3),
    'K-Nearest Neighbor': KNeighborsClassifier(n_jobs=-1, n_neighbors=205)
}


# Plot ROC using best params
lines = {}
for name, model in algos.items():
    train_roc_curve, test_roc_curve = md.build_roc_curve(ds_train_X, ds_train_y, ds_test_X, ds_test_y, model)
    lines[name] = test_roc_curve

# Plot ROC using best params
plt.plot_line(lines, 'optimized', 'Reveiver Operating Characteristic (ROC)', (8, 6), 'False Positive Rate', 'True Positive Rate', 'AUC')