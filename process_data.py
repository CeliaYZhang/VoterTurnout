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
        X.append(d[:-1])   ## Remove the last column 'PES1'
        y = d[-1]
        if y == '1':
            Y.append(0.)
        else:
            Y.append(1.)
    X = np.asarray(X, dtype='float')
    Y = np.asarray(Y)
    labels = data[0][:-1]
    return X, Y, labels

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
        X.append(d)
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


# Manually drop columns which are irrelevant.
# Drop the columns by their label.
def dropColumns(X, Xtest, labels):
    dropLabels = ['id','HRMONTH','HRYEAR4']

    colInds = [labels.index(label) for label in dropLabels]
    X = np.delete(X, colInds, axis=1)
    Xtest = np.delete(Xtest, colInds, axis=1)
    labels = [label for label in labels if label not in dropLabels]
    return X, Xtest, labels


# Expand features which have multiple classes which do not correspond
# with anything linear.
# For example if the marital status column has values 1->married 2->single 3->
# divorced there is not reason why single should be the average between married 
# and divorced. Thus we expand this out to three separate feature columns.
def binarizeFeatures(X, Xtest, labels):
    # Categories which seem pretty important
    binLabels = [
    'HRHTYPE',    # type of household
    'GEREG',      # region of US
    'PEMARITL',   # marital status
    'PTDTRACE',   # race
    'PRWKSTAT',   # employment status
    ]

    # Categories which seem kinda important
    binLabels.extend(
    [
    'GESTCEN',    # state
    'PRFAMTYP',   # family types
    'PRPERTYP',   # type of person based on armed forces participation
    'PREMPNOT',   # type of unemployed
    'PUIO1MFG',   # type of job
    ])

    # These seem kind of ridiculous
    # binLabels.extend(
    # [
    # 'PENATVTY',   # country of birth
    # 'PRMARSTA',   # marital status based on armed forces participation
    # 'GTCBSAST',   # kind of city
    # 'PRDTHSP',    # hispanic origin
    # 'PUWK',       # did you do work this week?
    # ])

    # I should delete these at some point
    # binLabels.extend(
    # [
    # 'PUCHINHH',  # changes in household composition
    # 'PERRP',     # more complex relationship to other person
    # 'PRFAMREL',  # family relationship to other person
    # 'GESTFIPS',  # another state category lel
    # 'PEMLR',     # employment status again recode lol
    # 'PRCOW1'
    # 'PRDTCOW1'
    # 'PRDTIND1'
    # 'PRDTOCC1'
    # 'PRMJIND1'
    # 'PRMJOCGR'
    # 'PRCHLD'
    # 'PRIMIND1'])

    colInds = [labels.index(label) for label in binLabels]
    for label in binLabels:
        colInd = labels.index(label)
        lb = preprocessing.LabelBinarizer()

        # Do something dangerous. Binarize on both the X and Xtest together to
        # ensure all classes amongst all input data are found.
        trainLength = len(X)
        X = np.concatenate((X, Xtest), axis=0)
        X = np.concatenate((X, lb.fit_transform(X[:,colInd])), axis=1)

        # Add the new class labels
        for bin in lb.classes_:
            labels.append(label + "_" + str(bin))

        # Delete the old class labels and column
        X = np.delete(X, colInd, axis=1)
        del labels[colInd]

        # Separate the test set back out
        Xtest = X[trainLength:]
        X = X[:trainLength]

    return X, Xtest, labels


## Import and process data and return X, Y, Xtest. 
def processData(train_file, test_file, perc):
    X, Y, labels = getTrainData(train_file)
    Xtest = getTestData(test_file)

    X, Xtest, labels = dropColumns(X, Xtest, labels)
    X, Xtest, labels = binarizeFeatures(X, Xtest, labels)
    
    X, Xtest = featureSelect(X, Y, Xtest, perc)
    X, Xtest = imputeMissingVals(X, Xtest)
    X, Xtest = scale(X, Xtest)
    return X, Y, Xtest
    
if __name__ == "__main__":
    # using this to test process_data.py
    train_data = "../train_2008.csv"
    test_data = "../test_2008.csv"
    X, Y, Xtest = processData(train_data, test_data, 0.5)

