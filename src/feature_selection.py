import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, std
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, learning_curve
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

def validate_model(model, X, y):
    cv = KFold(n_splits=3, random_state=None, shuffle=False)
    scores = cross_validate(model, X, y, scoring=['accuracy', 'precision', 'recall'], n_jobs=-1, return_train_score=True, return_estimator=True, cv=cv)
    return scores

def build_models(model, min_features, max_features):
    models = dict()
    for i in range(min_features, max_features):
        rfe = RFE(estimator=LogisticRegression(max_iter=5000), n_features_to_select=i)
        model = model
        models[str(i)] = Pipeline(steps=[('s', StandardScaler()), ('f', rfe), ('m', model)])
    return models

def run(X, y):
    models = build_models(LogisticRegression(max_iter=5000), 2, 20)
    accuracy = list()
    precision = list()
    recall = list()
    features = list()
    for feature, model in models.items():
        scores = validate_model(model, X, y)
        print(scores)
        accuracy.append(mean(scores['train_accuracy']))
        precision.append(mean(scores['train_precision']))
        recall.append(mean(scores['train_recall']))
        features.append(feature)

    # Plot
    plt.plot(features, accuracy, label='Accuracy')
    plt.plot(features, precision, label='Precision')
    plt.plot(features, recall, label='Recall')
    plt.title('SVC - Logistic Regression RFE')
    plt.xlabel('No. Features')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('../assets/images/feature_selection.png')
    plt.clf()



