# External imports
from sklearn.feature_selection import SelectKBest, f_classif

# Selects k best features using ANOVA
def select_k_best(X_train, y_train, k):
    kbest = SelectKBest(score_func=f_classif, k=k)
    kbest.fit(X_train, y_train)
    columns = kbest.get_support(indices=True)
    scores = kbest.scores_
    return columns, scores