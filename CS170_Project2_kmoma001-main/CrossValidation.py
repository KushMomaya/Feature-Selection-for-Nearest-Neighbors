from NN import *

data = np.loadtxt('../CS170_Small_Data__60.txt')
X = data[:, 1:]
Y = data[:, 0]

def loocv(X, Y, set, feature):
    accuracy = 0
    
    

    return accuracy