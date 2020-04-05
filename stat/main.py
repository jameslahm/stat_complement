import numpy as np


def HotellingT2(X, mu):
    n = len(X)
    mean = X.mean(0)
    S = (X-mean).T.dot(X-mean)/(n-1)
    return n*(mean-mu).dot(np.linalg.inv(S)).dot((mean-mu).T)


def JacobiSolve(A, b, x,delta):
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

def sor(A,b,w,x):
    while(1):
        temp=x.copy()
        for i in range(0,len(x)):
            x[i]=(1-w)*x[i]+w*(b[i]-sum([A[i][j]*x[j] for j in range(0,i)])-sum([A[i][j]*x[j] for j in range(i+1,len(x))]))/A[i][i]
        print(x)
        print(temp)
        if(max(abs(temp-x))<10**(-2)):
            break
    return x