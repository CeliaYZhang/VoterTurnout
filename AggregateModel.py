import numpy as np 
import csv

prepend = 'csv\\usable_predictions\\'

files = ['AdaDT_pred.csv',
         'AdaRF_pred.csv',
         'BaggedDT_pred.csv',
         'KNeigh_100_pred.csv',
         'LinearSVC_pred.csv',
         'LR_pred.csv',
         'NaiveBayes_pred.csv',
         'RF_pred.csv',
         'Ridge_pred.csv'
         ]

scores = [0.78138,
          0.77663,
          0.76600,
          0.73200,
          0.77663,
          0.77875,
          0.71388,
          0.77788,
          0.77500
          ]

def ypred(filename):
    Ypred = []
    with open(prepend + filename, 'r') as csvfile:
        rd = csv.reader(csvfile, delimiter=',')
        for row in rd:
            if row[1] == '1': 
                Ypred.append(-1.)
            elif row[1] == '2':
                Ypred.append(1.)
    return np.asarray(Ypred)
            
def meanPred():
    tot = ypred(files[0]) * scores[0]
    for i in range(1, len(files)):
        tot = tot + (ypred(files[i]) * scores[i])
    tot = np.sign(tot)
    return tot

def write(filename):
    Ypred = meanPred()
    with open(filename, 'w') as csvfile:
        writ = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writ.writerow(['id', 'PES1'])
        for i in range(len(Ypred)):
            y = Ypred[i]
            if y == 1:
                writ.writerow([i, 2])
            else:
                writ.writerow([i, 1])
        

write(prepend + 'Aggregate_Pred.csv')