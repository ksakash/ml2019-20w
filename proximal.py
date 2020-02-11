import numpy as np
import random

alpha = 1 # learning rate

Z = np.loadtxt("train")

X_train = Z[:600,1:] # training data
Y_train = Z[:600,0] # training labels

X_test = Z[600:,1:] # testing data
Y_test = Z[600:,0] # testing data

print ("dataset loaded")

# gradient for l2 loss
def getgradientL2(w, B):
    
    X_, Y_ = make_batch (X_train, Y_train, B)
    error = X_.dot(w) - Y_
    return 2*(np.transpose(X_).dot(error))

# to check convergence
def check_convergence(w_1, w_2, epsilon):
    if np.linalg.norm(w_1 - w_2, 2) < epsilon:
        return True
    return False

def make_batch (X, Y, B):
    n = Y.size
    B_eff = min (n, B)

    samples = random.sample(range(0, n), B_eff)
    X_ = X[samples,:]
    Y_ = Y[samples]

    return X_, Y_

# proximal function for L1 norm
def proximal (w, lamda):

    x = np.sign(w)*np.maximum(np.absolute(w) - lamda, 0)

    return x

def update_alpha(alpha, i):
    return alpha/(i+1)

# contains template for gardient descent
def gradient_descent():

    d = X_train.shape[1]
    print('number of dimension: ', d)
    w = np.zeros(d)
    B = 5
    lamda = 1

    for i in range (10):
        w_prev = w
        delta = getgradientL2(w, B)        
        w = w - alpha*delta
        print (w[:10])
        w = proximal (w, lamda)
        global alpha
        alpha = update_alpha (alpha, i)

    print (w)

gradient_descent()
