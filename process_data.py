import csv
import numpy as np
from sklearn import preprocessing 
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

## Grab the data from the file filename. Returns a matrix
## X with input data and a vector Y of 1 for positive response
## and 0 for negative response. All values are floats and
## np arrays are used. 
def getTrainData(filename):
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f)
        data = [d for d in data_iter] 
    X = []
    Y = []
    for i in range(1, len(data)):
        d = data[i]
        X.append(d[3:-1])   ## Remove the first 3 columns
        y = d[-1]
        if y == '1':
            Y.append(0.)
        else:
            Y.append(1.)
    X = np.asarray(X, dtype='float')
    Y = np.asarray(Y)
    return X, Y

## Grab the data from the file filename. Returns a matrix
## X with input data. All values are floats and
## np arrays are used. 
def getTestData(filename):
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f)
        data = [d for d in data_iter] 
    X = []
    for i in range(1, len(data)):
        d = data[i]
        X.append(d[3:])   ## Remove the first 3 columns
    X = np.asarray(X, dtype='float')
    return X

## Scales X to have 0 mean and 1 standard deviation, 
## and scales Xtest according to X. Returns the two
## scaled matrices. 
def scale(X, Xtest):
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)    
    Xtest = scaler.transform(Xtest)
    return X, Xtest

## Imputes missing values indicated by -1 in matrices
## X and Xtest as the mean of the columns of the missing
## values. 
def imputeMissingVals(X, Xtest):
    imp = preprocessing.Imputer(missing_values=-1, strategy='median', axis=0)
    imp.fit(X)
    X = imp.transform(X)
    Xtest = imp.transform(Xtest)
    return X, Xtest


## Select against featuers with low variance and select 
## a percentile perc of useful features. 
def featureSelect(X, Y, Xtest, perc):
    sel = VarianceThreshold()
    X = sel.fit_transform(X)
    Xtest = sel.transform(Xtest)
    sp = SelectPercentile(percentile=perc)
    X = sp.fit_transform(X, Y)
    Xtest = sp.transform(Xtest)
    return X, Xtest
    
## Import and process data and return X, Y, Xtest. 
def processData(train_file, test_file, perc):
    X, Y = getTrainData(train_file)
    Xtest = getTestData(test_file)
    X, Xtest = featureSelect(X, Y, Xtest, perc)
    X, Xtest = imputeMissingVals(X, Xtest)
    X, Xtest = scale(X, Xtest)
    return X, Y, Xtest
    
    
    