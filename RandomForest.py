import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import process_data as pd

trainfile08 = "train_2008.csv"
testfile08 = "test_2008.csv"
perc = 75

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
parameters = {'n_estimators': [100], 
              'max_features': ['auto', 'log2', None],
              'min_samples_leaf': [1,5, 10, 25, 50, 75, 100, 150]
              }

def classifyAll():
    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc, parameters, verbose=20)
    clf = clf.fit(X, Y)
    with open('GS_RandomForest_Results_2.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        for key, value in clf.cv_results_.items():
           writer.writerow([key, value])
    print(clf.cv_results_)
    return clf

def classifyOne(n_est, max_f, min_sl):
    clf = RandomForestClassifier(n_estimators=n_est, 
                                 max_features=max_f, min_samples_leaf=5)
    clf = clf.fit(X, Y)
    return clf

def predict():
    clf = classifyOne(500, None, 5)
    Ypred = clf.predict(Xtest)
    return Ypred

def write(filename):
    Ypred = predict()
    with open(filename, 'w') as csvfile:
        writ = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writ.writerow(['id', 'PES1'])
        for i in range(len(Ypred)):
            if Ypred[i] == 1:
                writ.writerow([i, 2])
            else:
                writ.writerow([i, 1])

write('predictions\GridRandomForestPred_2.csv')