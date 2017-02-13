import csv
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV, RFE

import process_data as pd

trainfile08 = "train_2008.csv"
testfile08 = "test_2008.csv"
testfile12 = "test_2012.csv"
perc = 100

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
_, _, Xtest2 = pd.processData(trainfile08, testfile12, perc)
print("Data processed")

def classifyRFE():
    clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(
            max_depth=1,
            max_features='log2'),
        n_estimators=750    
        )
    print("Performing RFECV") 
    selector = RFECV(clf, verbose=100, step=10)
    selector = selector.fit(X, Y)
    
    # Print results to file
    with open("logs\\adaBDT_rfe_results_3.txt", "w") as fle:
        fle.write("Grid Scores\n %s \n" % selector.grid_scores_)
        fle.write("Support\n %s \n" % selector.support_)
        fle.write("n_features\n %s \n" % selector.n_features_)
        
    # Transform X and refit with full data set
    print("Refitting X transformed")
    X_t = selector.transform(X)
    clf.fit(X_t, Y)
    return clf, selector

def predict(filename):
    clf, selector = classifyRFE()
    print("Predicting")
    Xtest_t = selector.transform(Xtest)
    Ypred = clf.predict(Xtest_t)
    write(filename, Ypred)
    return Ypred

def write(filename, Ypred):
    with open(filename, 'w') as csvfile:
        writ = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writ.writerow(['id', 'PES1'])
        for i in range(len(Ypred)):
            y = Ypred[i]
            if y == 1:
                writ.writerow([i, 2])
            else:
                writ.writerow([i, 1])

def transform():
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
    print(data)
    for i in range(len(data)):
        if not data[i]:
            colInds.append(i)
    X_t = np.delete(X, colInds, axis=1)
    Xtest_t = np.delete(Xtest, colInds, axis=1)
    Xtest2_t = np.delete(Xtest2, colInds, axis=1)
    return X_t, Xtest_t, Xtest2_t



def classifyClean():
    print("Classifying clean")
    clf = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(
                max_depth=1,
                max_features='log2'),
            n_estimators=750    
            )   
    selector = RFE(clf, n_features_to_select=790, step=10, verbose=100)
    selector = selector.fit(X, Y)
    
    with open("logs\\adaBDT_rfe_get_2012.txt", "w") as fle:
        fle.write("Support\n %s \n" % selector.support_)    
    
    # Transform X and refit with full data set
    print("Refitting X transformed")
    X_t = selector.transform(X)
    clf.fit(X_t, Y)
    return clf, selector

def predictClean():
    clf, selector = classifyClean()
    print("Predicting clean")
    Xtest_t = selector.transform(Xtest)
    Xtest2_t = selector.transform(Xtest2)
    Ypred = clf.predict(Xtest_t)
    Ypred2 = clf.predict(Xtest2_t)
    write('predictions\\adaBDT_2008_check.csv', Ypred)
    write('predictions\\adaBDT_2012_Pred.csv', Ypred2)
    
#X, Xtest, Xtest2 = transform()
predictClean()

#write('predictions\AdaDT_RFE_Pred_3.csv')