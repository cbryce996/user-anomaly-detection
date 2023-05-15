import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from numpy import mean
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, f1_score, roc_auc_score

# Fit model
def fit_model(model, X, y):
    return model.fit(X, y)

# Predict model and score using loss, accuracy, precision, recall, f1
def validate_model(model, X, y):
    y_pred = model.predict(X)
    prob = model.predict_proba(X)
    pred = prob[:,1]
    fp, tp, _ = roc_curve(y, pred)
    score = {
        'loss': np.round(log_loss(y, y_pred), 3),
        'accuracy': np.round(accuracy_score(y, y_pred), 3),
        'precision': np.round(precision_score(y, y_pred), 3),
        'recall': np.round(recall_score(y, y_pred), 3),
        'f1': np.round(f1_score(y, y_pred), 3),
        'roc': {
            'fp': fp,
            'tp': tp
        },
        'auc': round(roc_auc_score(y, pred), 3),
        'matrix': confusion_matrix(y, y_pred)
    }
    return score

# Select K best features using ANOVA coef
def select_k_best(X, y, k):
    kbest = SelectKBest(score_func=f_classif, k=k)
    kbest.fit(X, y)
    scores = {
        'columns': kbest.get_support(indices=True),
        'scores': kbest.scores_
    }
    return scores

# Build pipeline for model
def build_pipeline(algo):
    return Pipeline(steps=[
        ('scale', StandardScaler()),
        ('algo', algo)
    ])