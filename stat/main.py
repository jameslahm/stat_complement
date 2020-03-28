import numpy as np


def HotellingT2(X, mu):
    n = len(X)
    mean = X.mean(0)
    S = (X-mean).T.dot(X-mean)/(n-1)
    return n*(mean-mu).dot(np.linalg.inv(S)).dot((mean-mu).T)


def JaccobiSolve(A, b, x,delta):
    L = np.zeros(len(A), len(A))
    for i in range(len(A)):
        L[i][i] = A[i][i]
    B = np.linalg.inv(L).dot(L-A)
    while(True):
        xn = B.dot(x)+np.linalg.inv(L).dot(b)
        if(max(abs(xn-x)) < delta):
            break
        x = xn
    return x
