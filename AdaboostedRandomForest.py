import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

import process_data as pd

trainfile08 = "train_2008.csv"
testfile08 = "test_2008.csv"
perc = 75

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
parameters = {'base_estimator__min_samples_leaf': [1, 2, 5, 10, 25],
              'base_estimator__max_depth': [3, 4, 5, None],
              'base_estimator__max_features': ['auto', None]
              }
#parameters = {'base_estimator__min_samples_leaf': [1, 2, 5, 10, 25],
#              'base_estimator__max_depth': [1, 2, 3, 4, 5, None],
#              'base_estimator__max_features': ['auto', None]
#              }

def classifyAll():
    adc = AdaBoostClassifier(base_estimator=RandomForestClassifier(),
                             n_estimators=500)
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
    clf = classifyAll()
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

classifyAll()
print("Done")
#write('AdaRF_Pred_Big.csv')