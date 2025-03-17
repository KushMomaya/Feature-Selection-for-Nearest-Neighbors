import numpy as np
from CrossValidation import *

data = np.loadtxt('../CS170_Small_Data__60.txt')
X = data[:, 1:]
Y = data[:, 0]

def forward_search(X, Y):
    current_set = []

    for i in range(len(X[0])):
        print("On the " + str(i+1) + "th level of the search tree")
        feature_to_add = None
        best_sofar_accuracy = 0

        for k in range(len(X[0])):
            if k in current_set:
                continue

            print("Considering adding feature " + str(k+1))
            accuracy = loocv(X, Y, current_set, k)

            if accuracy > best_sofar_accuracy:
                best_sofar_accuracy = accuracy
                feature_to_add = k

        current_set.append(feature_to_add)
        print("On level " + str(i+1) + " I added feature " + str(feature_to_add+1) + " to the current set")
        print("Accuracy is " + str(best_sofar_accuracy))


forward_search(X, Y)