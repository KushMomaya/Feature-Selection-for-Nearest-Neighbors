import numpy as np

data = np.loadtxt('../CS170_Small_Data__60.txt')
X = data[:, 1:]
Y = data[:, 0]

def KNN(X, Y, point, k):
    distances = np.sqrt(np.sum((X - point) ** 2, axis=1))
    k_nearest = np.argsort(distances)[:k]
    k_nearest_labels = Y[k_nearest]

    unique, counts = np.unique(k_nearest_labels, return_counts=True)
    most_common_label = unique[np.argmax(counts)]
    return most_common_label

print(KNN(X, Y, X[44], 3))