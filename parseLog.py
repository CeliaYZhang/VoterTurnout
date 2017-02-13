import numpy as np 

def getFile(filename):
    data = []
    for line in open(filename, 'r'):
        data.append(line.split(','))
    return data

def parse_adaBDT():
    data = getFile("logs\\adaBDT_1_half.txt")[1:]
    
    ## Remove surrounding words and symbols
    parsed = []
    for i in range(len(data)):
        dat = data[i]
        if i % 2 == 0:
            continue
        line = []
        for j in range(len(dat)-1):
            d = dat[j]
            val = d.split('=')[1]
            if j == 1:
                if val == 'sqrt':
                    line.append(1.)
                elif val == 'log2':
                    line.append(2.)
                elif val == 'None':
                    line.append(3.)
                else:
                    line.append(-1)
            else:
                line.append(float(val))
        parsed.append(line)   
    return parsed

def takeMeans(arr, numlines=3):
    means = []
    for i in range(len(arr) / 3):
        mean = (arr[3*i][3] + arr[3*i+1][3] + arr[3*i+2][3]) / 3.
        means.append(arr[3*i][:3] + [mean])
    return means


def parse_RF():
    data = getFile("logs\\RF_log_2_part.txt")[1:]
    
    ## Remove surrounding words and symbols
    parsed = []
    for i in range(len(data)):
        dat = data[i]
        if i % 2 == 0:
            continue
        line = []
        for j in range(len(dat)-1):
            d = dat[j]
            val = d.split('=')[1]
            if j == 1:
                if val == 'auto':
                    line.append(1.)
                elif val == 'log2':
                    line.append(2.)
                elif val == 'None':
                    line.append(3.)
                else:
                    line.append(-1)
            else:
                line.append(float(val))
        parsed.append(line)   
    return parsed

def parse_ABRF():
    data = getFile("logs\\adaBRF.txt")[1:]
    ## Remove surrounding words and symbols
    parsed = []
    for i in range(len(data)):
        dat = data[i]
        if i % 2 == 0:
            continue
        line = []
        for j in range(len(dat)-1):
            d = dat[j]
            val = d.split('=')[1]
            if j == 2:
                if val == 'auto':
                    line.append(1.)
                elif val == 'log2':
                    line.append(2.)
                elif val == 'None':
                    line.append(3.)
                else:
                    line.append(-1)
            else:
                line.append(float(val))
        parsed.append(line)   
    return parsed

for p in takeMeans(parse_ABRF()):
    print p

#for p in takeMeans(parse_adaBDT()):
#    print p

#for p in takeMeans(parse_RF()):
#    print p