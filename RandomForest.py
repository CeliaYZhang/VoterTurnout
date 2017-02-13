import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import process_data as pd

trainfile08 = "train_2008.csv"
testfile08 = "test_2008.csv"
testfile12 = "test_2012.csv"
perc = 100

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
_, _, Xtest2 = pd.processData(trainfile08, testfile12, perc)
#parameters = {'n_estimators': [500], 
#              'min_samples_leaf': [5, 10]
#              #'max_features': ['auto', 'log2', None],
#              #'min_samples_leaf': [1,5, 10, 25, 50, 75, 100, 150]
#              }

def transform(X, Xtest):
    data = []
    for line in open("logs\\best_features.txt", 'r'):
        for val in line.split():
            if val == 'True':
                data.append(True)
            elif val == 'False':
                data.append(False)
            else:
                data.append("NO")
    colInds = []
    for i in range(len(data)):
        if data[i]:
            colInds.append(i)
    X = np.delete(X, colInds, axis=1)
    Xtest = np.delete(Xtest, colInds, axis=1)
    return X, Xtest

def classifyAll():
    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc, parameters, verbose=20)
    clf = clf.fit(X, Y)
    with open("RF_slimmed_5vs7.csv", 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        for key, value in clf.cv_results_.items():
            writer.writerow([key, value])
    print(clf.cv_results_)
    return clf

def classifyOne(n_est, max_f, min_sl):
    clf = RandomForestClassifier(n_estimators=n_est, 
                                 max_features=max_f, min_samples_leaf=min_sl)
    clf = clf.fit(X, Y)
    return clf

def predict():
    clf = classifyOne(500, 'auto', 1)
    Ypred = clf.predict(Xtest)
    Ypred2 = clf.predict(Xtest2)
    write('predictions\\RF_500_auto_1_2008_check.csv', Ypred)
    write('predictions\\RF_500_auto_1_2012_Pred.csv', Ypred2)

def write(filename, Ypred):
    with open(filename, 'w') as csvfile:
        writ = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writ.writerow(['id', 'PES1'])
        for i in range(len(Ypred)):
            if Ypred[i] == 1:
                writ.writerow([i, 2])
            else:
                writ.writerow([i, 1])

predict()