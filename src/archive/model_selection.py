from sklearn.metrics import accuracy_score

# parameter = {'C': [0.1, 1, ..., 10]}
def param_search(X_train, y_train, X_test, y_test, classifier, parameter):
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
    
    return train_scores, test_scores