import csv
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import process_data as pd

trainfile08 = "train_2008.csv"
testfile08 = "test_2008.csv"
perc = 75

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
parameters = {'n_estimators': [50, 100, 200, 350, 500],
              'base_estimator__min_samples_leaf': [1, 5, 10, 25, 50, 75],
              'base_estimator__max_features': ['auto', 'log2', None]
              }

def classifyAll():
    bc = BaggingClassifier(base_estimator=DecisionTreeClassifier())
    clf = GridSearchCV(bc, parameters, verbose=10)
    clf = clf.fit(X, Y)
    print(clf.cv_results_)
    print(clf.best_params_)
    return clf

def classifyOne():
    clf = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(
            max_depth=5,
            max_features=None), 
        n_estimators=500)    
    return clf

def predict():
    clf = classifyOne()
    clf = clf.fit(X, Y)
    Ypred = clf.predict(Xtest)
    return Ypred

def write(filename):
    Ypred = predict()
    with open(filename, 'w') as csvfile:
        writ = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writ.writerow(['id', 'PES1'])
        for i in range(len(Ypred)):
            y = Ypred[i]
            if y == 1:
                writ.writerow([i, 2])
            else:
                writ.writerow([i, 1])

write('predictions\BaggedDT_Pred.csv')