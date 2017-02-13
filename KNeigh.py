import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import process_data as pd

trainfile08 = "train_2008.csv"
testfile08 = "test_2008.csv"
perc = 75

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
print("Data processed")

parameters = {'n_neighbors': [250, 500, 1000], #[3, 5, 10, 25, 50, 100],
              #'weights': ['uniform', 'distance']
              }

def classifyAll():
    print("Begin classification")
    kn = KNeighborsClassifier()
    clf = GridSearchCV(kn, parameters, verbose=10)
    clf = clf.fit(X, Y)
    print(clf.cv_results_)
    print(clf.best_params_)
    return clf

def classifyOne():
    print("Begin classification")    
    clf = KNeighborsClassifier(n_neighbors=5)
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


classifyAll()
#write('predictions\KNeigh_Small_Pred.csv')
print("Done")