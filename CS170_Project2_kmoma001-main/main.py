import numpy as np
from FeatureSelection import *

def main():
    print("Welcome to Kush Momaya's Feature Selection Algorithm!")
    print("Would you like to test the algorithm on a small dataset or large dataset? (1 for small, 2 for large)")
    input_data = input()
    print("Would you like to use forward selection or backward elimination? (1 for forward, 2 for backward)")
    input_algo = input()
    if input_data == '1':
        data = np.loadtxt('../CS170_Small_Data__60.txt')
        print("You are testing on a small dataset")
    elif input_data == '2':
        data = np.loadtxt('../CS170_Large_Data__96.txt')
        print("You are testing on a large dataset")

    X = data[:, 1:]
    Y = data[:, 0]

    if input_algo == '1':
        print("Forward Selection Trace")
        forward_search(X, Y)
    elif input_algo == '2':
        print("Backward Elimination Trace")
        backward_search(X, Y)


main()