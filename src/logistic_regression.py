import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def run_logistic_regression(X_train, y_train, X_test, y_test):
    # Classification algorithm
    clf = LogisticRegression()

    # Cross validation
    cv = KFold(n_splits=5, random_state=None, shuffle=False)

    # Parameter search
    srch = GridSearchCV(estimator=clf, param_grid={"C":np.logspace(-3,3,7), "penalty":["l2"], "solver":["lbfgs"], "max_iter":[5000], "n_jobs":[-1]}, cv=cv)

    # Results
    rslt = srch.fit(X_train, y_train)

    # Predict
    y_pred = rslt.best_estimator_.predict(X_test)

    # Print
    print("Accuracy Score: %s" %accuracy_score(y_test, y_pred[:len(y_test)]))
    print("Classification Report: \n %s" %(classification_report(y_test, y_pred[:len(y_test)])))

    # Confusion Matrix
    #matrix = confusion_matrix(y, y_pred)