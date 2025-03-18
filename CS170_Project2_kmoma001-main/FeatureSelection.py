import numpy as np
from CrossValidation import *

#data = np.loadtxt('../CS170_Small_Data__60.txt')
#X = data[:, 1:]
#Y = data[:, 0]

def forward_search(X, Y):
    current_set = []

    for i in range(len(X[0])):
        print("On the " + str(i+1) + "th level of the search tree")
        feature_to_add = 0
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
        print("")
        print("On level " + str(i+1) + " I added feature " + str(feature_to_add+1) + " to the current set")
        print("")
        print("Accuracy is " + str(best_sofar_accuracy))
        print("")
        print("Current Set is " + str([j + 1 for j in current_set]))
        print("")




def backward_search(X, Y):
    current_set = np.array([i for i in range(len(X[0]))])

    for i in range(len(X[0])):
        print("On the " + str(i+1) + "th level of the search tree")
        feature_to_remove = None
        best_sofar_accuracy = 0

        for k in range(len(current_set)):
            print("Considering removing feature " + str(current_set[k]+1))
            accuracy = loocv(X, Y, np.delete(current_set, k), 0)

            if accuracy > best_sofar_accuracy:
                best_sofar_accuracy = accuracy
                feature_to_remove = k
        removed_feature = current_set[feature_to_remove]
        current_set = np.delete(current_set, feature_to_remove)
        print("")
        print("On level " + str(i+1) + " I removed feature " + str(removed_feature+1) + " from the current set")
        print("")
        print("Accuracy is " + str(best_sofar_accuracy))
        print("")
        print("Current Set is " + str([int(j + 1) for j in current_set]))
        print("")

#forward_search(X, Y)
#backward_search(X, Y)