import numpy as np
from NN import *

data = np.loadtxt('../CS170_Small_Data__60.txt')
X = data[:, 1:]
Y = data[:, 0]

def loocv(X, Y): #, set, feature):
    num_correct = 0
    accuracy = 0
    for i in range(len(X)):
        point = X[i]
        label = Y[i]
        nn_distance = float('inf')
        nn_location = None
        nn_label = None
        for k in range(len(X)):
            if k == i:
                continue
            distance = np.sqrt(np.sum((point - X[k]) ** 2))
            if distance < nn_distance:
                nn_distance = distance
                nn_location = k
                nn_label = Y[nn_location]
        #print("Point " + str(i+1) + " is class " + str(label) + " and its nearest neighbor is point " + str(nn_location+1) + " which is class " + str(nn_label))
        if label == nn_label:
            num_correct += 1

    accuracy = num_correct / len(X)
    return accuracy


print(loocv(X, Y))