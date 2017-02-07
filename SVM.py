import process_data as pd
import csv
from sklearn import svm
from sklearn.model_selection import GridSearchCV

trainfile08 = 'train_2008.csv'
testfile08 = 'test_2008.csv'
perc = 75

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
parameters = {'C': [0.01, 0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 10, 100]
              }

def classifyAll():
    model = svm.SVC()
    clf = GridSearchCV(model, parameters, verbose=20)
    clf = clf.fit(X, Y)
    with open('SVM.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        for key, value in clf.cv_results_.items():
           writer.writerow([key, value])
    print(clf.cv_results_)
    return clf

def classifyOne(c_err, gam):
    clf = svm.SVC(C=c_err, gamma=gam)
    clf = clf.fit(X, Y)
    return clf

def predict():
    clf = classifyOne(1, 1)
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

# write('predictions\SVM_pred.csv')
classifyAll()