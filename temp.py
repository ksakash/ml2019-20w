import numpy as np
import random

n = 5
m = 10
B = 5
X = np.zeros((n,m))
Y = np.zeros(m)
Z = np.copy(Y)

Z[1] = 1000

X[0][1] = 4
X[2][1] = 3
X[4][1] = 2

x = X[:,1]

Y[0] = 20
Y[1] = 2
Y[2] = 53
Y[3] = 32


print (X.dot(Y))
