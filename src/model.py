# External inputs
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

# Returns roc curves for train/test data
def build_roc_curve(X_train, y_train, X_test, y_test, classifier):
    # Fit model
    classifier.fit(X_train, y_train)

    y_train_pred = classifier.predict_proba(X_train)[:,1]
    fpr, tpr, _ = roc_curve(y_train, y_train_pred)
    train_roc_curve = {
        'X': fpr,
        'Y': tpr,
        'Label': round(roc_auc_score(y_train, y_train_pred), 3)
    }

    y_test_pred = classifier.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    test_roc_curve = {
        'X': fpr,
        'Y': tpr,
        'Label': round(roc_auc_score(y_test, y_test_pred), 3)
    }

    return train_roc_curve, test_roc_curve

# Searches parameters and returns accuracy scores on train/test data
def param_search(X_train, y_train, X_test, y_test, classifier, parameter):
    train_curve = {}
    test_curve = {}

    train_scores = []
    test_scores = []
    
    for name, values in parameter.items():
        for value in values:
            # Set params
            classifier.set_params(**{
                name: value
            })

            # Fit model
            classifier.fit(X_train, y_train)

            # Train score
            y_train_pred = classifier.predict(X_train)
            train_scores.append(round(accuracy_score(y_train, y_train_pred), 3))
            
            # Test score
            y_test_pred = classifier.predict(X_test)
            test_scores.append(round(accuracy_score(y_test, y_test_pred), 3))

        train_curve['X'] = values
        test_curve['X'] = values

    train_curve['Y'] = train_scores
    train_curve['Label'] = max(train_scores)

    test_curve['Y'] = test_scores
    test_curve['Label'] = max(test_scores)

    print(train_curve)
    
    return train_curve, test_curve