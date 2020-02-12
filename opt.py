import numpy as np
import random
import time
import math

#alpha = 1 # learning rate

Z = np.loadtxt("train")

X_train = Z[:,1:] # training data
Y_train = Z[:,0] # training labels

X_test = Z[:200,1:] # testing data
Y_test = Z[:200,0] # testing data

print ("dataset loaded")

# gradient for l2 loss
def getgradientL2(w):
    error = X_train.dot(w) - Y_train
    return 2*(np.transpose(X_train).dot(error))

def getgradientL1(w):
    return np.sign(w)

# to check convergence
def check_convergence(w_1, w_2, epsilon):
    if np.linalg.norm(w_1 - w_2, 2) < epsilon:
        return True
    return False

def make_batch (X, Y, B):
    n = Y.size
    B_eff = min (n, B)

    samples = random.sample(range(0, n), int(B_eff))
    X_ = X[samples,:]
    Y_ = Y[samples]

    return X_, Y_

def getMBgradient(w, B, lamda):
    X_, Y_ = make_batch (X_train, Y_train, B)
    error = X_.dot(w) - Y_
    L2_grad = 2*(np.transpose(X_).dot(error))
    L1_grad = np.sign(w)

    return L2_grad + lamda*L1_grad

# proximal function
def proximal (w, lamda):

    x = np.sign(w)*np.maximum(np.absolute(w) - lamda, 0)

    return x

def update_alpha(alpha, i):
    min_ = 1e-3
    if alpha <= min_:
        return min_
    return alpha/math.sqrt(i+1)

def get_loss (w):
    L2 = np.linalg.norm(X_train.dot(w) - Y_train, 2)
    L2 = L2*L2
    L1 = np.linalg.norm(w, 1)
    return L1 + L2

# contains template for gardient descent
def gradient_descent():

    d = X_train.shape[1]
    w = np.zeros(d)
    lamda = 1
    alpha = 0.1
    epsilon = 0.01

    curr_loss = 0
    prev_loss = 0

    for i in range (10000):
        delta = getgradientL2(w) + lamda*getgradientL1(w)
        w = w - alpha*delta
        alpha = update_alpha (alpha, i)
        if not i%10:
            curr_loss = get_loss(w)
            loss_diff = abs(curr_loss-prev_loss)
            print ("loss: ", curr_loss)
            prev_loss = curr_loss
            if loss_diff <= epsilon:
                print ("saturated")
                break

    print (np.linalg.norm(w))
    print (np.linalg.norm(X_test.dot(w) - Y_test, 2))
    w_t = np.loadtxt("wAstTrain")
    print (np.linalg.norm(w-w_t, 2))

def proximal_descent():

    d = 1000
    lamda = 0.125
    alpha = 0.1
    epsilon = 0.01

    for _ in range(1):
        w = np.zeros(d)
        curr_loss = 0
        prev_loss = get_loss(w)

        for i in range (100):

            delta = getgradientL2(w)
            w = w - alpha*delta
            w = proximal (w, lamda)
            curr_loss = get_loss(w)

            if not i%10:
                loss_diff = abs(curr_loss - prev_loss)
                print ("loss: ", curr_loss)
                prev_loss = curr_loss
                if loss_diff <= epsilon:
                    print("saturated")
                    break

    print(np.linalg.norm(w))
    print (np.linalg.norm((X_test.dot(w) - Y_test), 2))
    w_t = np.loadtxt("wAstTrain")
    print (np.linalg.norm((w - w_t),2))
    print (np.linalg.norm(w_t, 1))


def MBGD ():
    d = 1000
    w = np.zeros(d)
    lamda = 0.1
    alpha = 0.1
    epsilon = 0.0001
    B = 75

    curr_loss = 0
    prev_loss = get_loss(w)

    for i in range (15000):
        delta = getMBgradient(w, B, lamda)
        w = w - alpha*delta
        alpha = update_alpha (alpha, i)
        curr_loss = get_loss (w)

        if not i%10:
            loss_diff = abs(curr_loss - prev_loss)
            print ("loss: ", curr_loss)
            prev_loss = curr_loss
            if loss_diff <= epsilon:
                print ("saturated")
                break

    print (w)


def soft_gradient (rho, lamda):
    if (rho + lamda < 0):
        return rho + lamda
    elif (rho - lamda > 0):
        return rho - lamda
    else:
        return 0

def coordinate_descent():
    d = 1000
    w = np.zeros(d)
    lamda = 0.35
    epsilon = 0.001
    loss = 0

    for i in range (5000):
        i_ = i%d
        rho = np.dot(X_train[:,i_], Y_train - np.dot(X_train, w) + w[i_]*X_train[:,i_])
        z = np.linalg.norm(X_train[:,i_], 2)**2

        w[i_] = soft_gradient(rho, lamda)/z

        loss = get_loss(w)

    print (w)
    print (loss)

tic = time.perf_counter()
# gradient_descent()
# proximal_descent()
# MBGD()
# coordinate_descent()
toc = time.perf_counter()

print ("time taken: ", toc-tic)
