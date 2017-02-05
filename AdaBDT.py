import csv
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import process_data as pd

trainfile08 = "train_2008.csv"
testfile08 = "test_2008.csv"
perc = 75

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
parameters = {'n_estimators': [50, 100, 250, 500, 750, 1000],
              'base_estimator__max_depth': [5],
              'base_estimator__max_features': [None]
              }

def classifyAll():
    adc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    clf = GridSearchCV(adc, parameters, verbose=10)
    clf = clf.fit(X, Y)
    print(clf.cv_results_)
    print(clf.best_params_)
    return clf

def classifyOne():
    clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(
            max_depth=1,
            max_features=None), 
        n_estimators=250)    
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

write('predictions\AdaDT_Pred.csv')