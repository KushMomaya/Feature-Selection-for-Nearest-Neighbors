import numpy as np

data = np.loadtxt('../CS170_Small_Data__60.txt')
X = data[:, 1:]
Y = data[:, 0]

def NN(X, Y, point):
    distances = np.sqrt(np.sum((X - point) ** 2, axis=1))
    nearest = np.argmin(distances)
    nearest_label = Y[nearest]
    return nearest_label
