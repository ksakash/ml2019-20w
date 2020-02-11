import numpy as np
import random
import math
import time as tm

Z = np.loadtxt("train")

X_train = Z[:,1:]
Y_train = Z[:,0]

X_test = Z[600:,1:]
Y_test = Z[600:,0]

print ("data loaded")

# calculate gradient descent
def getgradientL1(w):
    return np.sign(w)

# gradient for l2 loss
def getgradientL2(w, B):
    
    X_, Y_ = make_batch (X, Y, B)
    error = X_.dot(w) - Y_
    grad = 2*(np.transpose(X_).dot(w))
    return grad

# to check convergence
def check_convergence(w_1, w_2, epsilon):
    if abs(w_1 - w_2) < epsilon:
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

    x = np.zeros(len(w))
    x = np.sign(w)*np.maximum(np.absolute(w) - lamda, 0)

    return x

# for random coord choose a random int between 0 to d-1
def get_random_coord (d):
    i = random.randint(0, d-1)
    return i

# for cyclic coord increment by one
def get_cyclic_coord (curr, d):
    curr += 1
    curr = curr%d
    return curr

# for rand perm coord choose a random permutation and choose from it
def get_rand_perm_coord (curr, perm, d):
    if curr >= d-1 or curr < 0:
        curr = 0
        perm = np.random.permutation(d)
    else:
        curr += 1

    return (curr, perm)

def get_coord_gradient_L1 (w, idx):
    return np.sign(w[idx])

def get_coord_gradient_L2 (w, idx):
    error = X_train.dot(w) - Y_train
    x = X_train[:,idx]
    grad = error.dot(x)
    return 2*grad

def update_aplha(alpha, i):
    return alpha/math.sqrt(i+1)

def get_loss (w):
    L2 = np.linalg.norm(X_train.dot(w) - Y_train, 2)
    L2 = L2*L2
    L1 = np.linalg.norm(w, 1)
    return L1 + L2

# contains template for gardient descent
def gradient_descent():

    d = X_train.shape[1]
    # print (d)
    w = np.zeros(d)
    # print(w.shape)
    B = 5
    lamda = 1
    idx = 0
    curr = 0
    alpha = 0.1
    epsilon = 0.01 # 0.01

    id_count = 0

    loss_prev_coord = get_loss (w)
    loss_curr_coord = get_loss (w)
    C = 2
    C_1 = 1

    perm = np.random.permutation(d)
    w_idx = 0
    loss_prev = 0
    loss_curr = 0

    tic = tm.perf_counter()

    for i in range (30000):
        #w_prev = np.copy(w)
        
        delta1 = get_coord_gradient_L1(w, idx)
        delta2 = get_coord_gradient_L2(w, idx)
        delta = C_1*delta1 + C*delta2
        w[idx] = w[idx] - alpha*delta

        loss_curr = get_loss (w)

        loss_diff = abs (loss_curr - loss_prev)
        loss_prev = loss_curr

        if loss_diff <= epsilon:
            #print ("idx ", idx, " converged")
            loss_curr_coord = loss_curr
            #print ("loss: ", loss_curr_coord)
            total_diff = (loss_curr_coord - loss_prev_coord)
            if (total_diff >= 0):
                #print ("no change")
                #w[idx] = w_idx
                if abs(w[idx]) < 0.01:
                    w[idx] = 0
                #perm = np.delete(perm, curr, 0)
                #print (perm.size)
            loss_prev_coord = loss_curr_coord
            print ("total diff: ", total_diff)
            if abs(w[idx]) < 0.1:
                w[idx] = 0
            curr, perm = get_rand_perm_coord (curr, perm, d)
            idx = perm[curr]
            
            #idx = get_cyclic_coord (idx, d)
            #idx = get_random_coord (d)
            w_idx = w[idx]
            alpha = 0.01
            #print ("idx count: ", id_count)
            id_count = 0
            #continue
            
        id_count += 1
        
        #print ("iteration ", i)
        #if not i%1:
            # print ("loss: ", get_loss(w))
        alpha = update_aplha (alpha, id_count)
    
    toc = tm.perf_counter()
    print (w)
    print ("final loss: ", get_loss(w))
    print ("time taken: ", toc-tic)

gradient_descent()
