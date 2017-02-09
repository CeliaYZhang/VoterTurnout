import process_data as pd
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

trainfile08 = 'train_2008.csv'
testfile08 = 'test_2008.csv'
perc = 75

X, Y, Xtest = pd.processData(trainfile08, testfile08, perc)
parameters = {'hidden_layer_sizes': [(10, 10), (30, 30), (30, 10), (50, 50), (50, 30), (50, 10), (100, 100), (100, 50), (100, 10), (10, 10, 10)],
              'alpha': [0.0001, 0.01, 1, 100]
              }

def classifyAll():
    model = MLPClassifier()
    clf = GridSearchCV(model, parameters, verbose=20)
    clf = clf.fit(X, Y)
    with open('NN.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        for key, value in clf.cv_results_.items():
           writer.writerow([key, value])
    print(clf.cv_results_)
    return clf

def classifyOne(hidden_layer_sizes, alpha):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
    clf = clf.fit(X, Y)
    return clf

def predict():
    clf = classifyAll()
    # clf = classifyOne((10, 10), 0.1)
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

write('csv/NN_pred.csv')