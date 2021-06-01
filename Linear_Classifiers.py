# import scipy.io as sio
import numpy as np
import math
import argparse
import scipy.io as sio

def data_loader(data_name):
    if data_name == "starplus":
        
        data = np.load('Starplus.npz')
        X_train = data['X_train'] / 10.0
        Y_train = data['Y_train']
        X_test = data['X_test'] / 10.0
        Y_test = data['Y_test']
        return np.transpose(X_train), Y_train.reshape(-1, 1), np.transpose(X_test), Y_test.reshape(-1, 1)

    else:
        X_train = [[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [4.0, 4.0, 1.0], [5.0, 5.0, 1.0], [1.0, 3.0, 1.0], [2.0, 4.0, 1.0], [4.0, 6.0, 1.0], [5.0, 7.0, 1.0]]
        X_test = [[3.0, 3.0, 1.0], [3.0, 5.0, 1.0]]
        Y_train = [[1.0], [1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [-1.0]]
        Y_test = [[1.0], [-1.0]]
        return np.transpose(np.asarray(X_train)), np.asarray(Y_train), np.transpose(np.asarray(X_test)), np.asarray(Y_test)

def sigmoid(a):
    return 1/(1 + np.exp(-a))

def Logisitc_Regression(X, Y, learningRate=0.01, maxIter=100):
    """
    Input:
        X: a (D+1)-by-N matrix (numpy array) of the input data; that is, we have concatenate "1" for you
        Y: a N-by-1 matrix (numpy array) of the label
    Output:
        w: the linear weight vector. Please represent it as a (D+1)-by-1 matrix (numpy array).
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
    """
    N = X.shape[1] # shape returns (num arrays, length arrays)
    D_plus_1 = X.shape[0]
    w = np.zeros((D_plus_1, 1))
    Y[Y == -1] = 0.0 # change label to be {0, 1}

    for t in range(maxIter):
        w = np.transpose(np.transpose(w) - learningRate * (np.zeros((1, D_plus_1)) + sum([((Y[n, :] - sigmoid(np.dot(w.T, X[:, n]))).item() * X[:, n]) for n in range(N)])) * -1/N)

    Y[Y == 0] = -1  # change label to be {-1, 1}
    return w


def Perceptron(X, Y, learningRate=0.01, maxIter=100):
    """
    Input:
        X: a (D+1)-by-N matrix (numpy array) of the input data; that is, we have concatenate "1" for you
        Y: a N-by-1 matrix (numpy array) of the label; labels are {-1, 1} and you have to turn them to {0, 1}
    Output:
        w: the linear weight vector. Please represent it as a (D+1)-by-1 matrix (numpy array).
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
        3. np.sign: for sign
    """
    N = X.shape[1]
    D_plus_1 = X.shape[0]
    w = np.zeros((D_plus_1, 1))
    np.random.seed(1)

    for t in range(maxIter):
        permutation = np.random.permutation(N)
        X = X[:, permutation]
        Y = Y[permutation, :]
        for n in range(N):
            if np.sign(np.dot(w.T, X[:, n])) != Y[n, :]:
                w = np.transpose(np.transpose(w) + X[:, n] * learningRate * Y[n, :])
    return w


def Accuracy(X, Y, w):
    Y_hat = np.sign(np.matmul(X.transpose(), w))
    correct = (Y_hat == Y)
    return float(sum(correct)) / len(correct)


def main(args):
    X_train, Y_train, X_test, Y_test = data_loader(args.data)
    print("number of training data instances: ", X_train.shape)
    print("number of test data instances: ", X_test.shape)
    print("number of training data labels: ", Y_train.shape)
    print("number of test data labels: ", Y_test.shape)

    if args.algorithm == "logistic":
    #----------------Logistic Loss-----------------------------------
        w = Logisitc_Regression(X_train, Y_train,  maxIter=100, learningRate=0.1)
    # ----------------Perceptron-----------------------------------
    else:
        w = Perceptron(X_train, Y_train, maxIter=100, learningRate=0.1)

    training_accuracy = Accuracy(X_train, Y_train, w)
    test_accuracy = Accuracy(X_test, Y_test, w)
    print("Accuracy: training set: ", training_accuracy)
    print("Accuracy: test set: ", test_accuracy)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running linear classifiers")
    parser.add_argument('--algorithm', default="logistic", type=str)
    parser.add_argument('--data', default="simple", type=str)
    args = parser.parse_args()
    main(args)
