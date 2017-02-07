import process_data as pd
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

trainfile08 = 'train_2008.csv'
testfile08 = 'test_2008.csv'
perc = 75

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
parameters = {}

def classifyAll():
    model = GaussianNB()
    clf = GridSearchCV(model, parameters, verbose=20)
    clf = clf.fit(X, Y)
    with open('NaiveBayes.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        for key, value in clf.cv_results_.items():
           writer.writerow([key, value])
    print(clf.cv_results_)
    return clf

def classifyOne():
    clf = GaussianNB(priors=None)
    clf = clf.fit(X, Y)
    return clf

def predict():
    clf = classifyOne()
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

write('NaiveBayes_pred.csv')