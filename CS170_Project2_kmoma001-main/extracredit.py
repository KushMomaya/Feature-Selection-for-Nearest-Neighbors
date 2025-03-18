import numpy as np
from FeatureSelection import *
import matplotlib.pyplot as plt

data = np.genfromtxt('../wine/wine.data', delimiter=',')
X = data[:, 1:]
Y = data[:, 0]

def normalize(X):
    for i in range(len(X[0])):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        #print(X[:, i])
    return X

def graph_features(X, Y):
    forward_accuracies, forward_features = forward_search(X, Y)

    plt.figure(figsize=(10, 5))

    plt.bar(range(len(forward_accuracies)), forward_accuracies, color='b')
    plt.xlabel('Feature Set')
    plt.ylabel('Accuracy')
    plt.title('Forward Selection')
    #print(len(forward_accuracies))
    #print(len(forward_features))
    plt.xticks(range(len(forward_accuracies)), [str([int(j + 1) for j in fset]) for fset in forward_features])

    plt.tight_layout()
    plt.show()




normX = normalize(X)
graph_features(X, Y)