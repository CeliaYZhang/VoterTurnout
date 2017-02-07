import process_data as pd
import csv
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

trainfile08 = 'train_2008.csv'
testfile08 = 'test_2008.csv'
perc = 75

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
parameters = {'loss': ['log', 'hinge'],
              'alpha': [0.01],
              'penalty': ['l2', 'l1', 'elasticnet']
              }

def classifyAll():
    model = SGDClassifier()
    clf = GridSearchCV(model, parameters, verbose=20)
    clf = clf.fit(X, Y)
    with open('SGDLog.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        for key, value in clf.cv_results_.items():
           writer.writerow([key, value])
    print(clf.cv_results_)
    return clf

def classifyOne(loss_fun, a, pen):
    clf = SGDClassifier(loss=loss_fun, alpha=a, penalty=pen)
    clf = clf.fit(X, Y)
    return clf

def predict():
    clf = classifyOne('log', 0.01, 'elasticnet')
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

write('SGDLog_pred.csv')