import numpy as np
import matplotlib.pyplot as plt
from FeatureSelection import *
import time

#data = np.loadtxt('../CS170_Small_Data__60.txt')
data = np.loadtxt('../CS170_Large_Data__96.txt')
X = data[:, 1:]
Y = data[:, 0]

def graph_features(X, Y):
    start = time.time()
    forward_accuracies, forward_features = forward_search(X, Y)
    end = time.time()
    print("Time taken to run: " + str(end - start) + " seconds")
    backward_accuracies, backward_features = backward_search(X, Y)
    forward_accuracies = forward_accuracies[:3] + forward_accuracies[-3:]
    forward_features = forward_features[:3] + forward_features[-3:]
    backward_accuracies = backward_accuracies[:3] + backward_accuracies[-3:]
    backward_features = backward_features[:3] + backward_features[-3:]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(forward_accuracies)), forward_accuracies, color='b')
    plt.xlabel('Feature Set')
    plt.ylabel('Accuracy')
    plt.title('Forward Selection')
    #print(len(forward_accuracies))
    #print(len(forward_features))
    plt.xticks(range(len(forward_accuracies)), [str([int(j + 1) for j in fset]) for fset in forward_features])

    plt.subplot(1, 2, 2)
    plt.bar(range(len(backward_accuracies)), backward_accuracies, color='r')
    plt.xlabel('Feature Set')
    plt.ylabel('Accuracy')
    plt.title('Backward Elimination')
    plt.xticks(range(len(backward_accuracies)), [str([int(j + 1) for j in fset]) for fset in backward_features])

    plt.tight_layout()
    plt.show()

graph_features(X, Y)