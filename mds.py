#%%
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt

def d_square_matrix_function(x):
    x_t=x.T
    x_matrix=np.dot(x_t,x)
    ones_matrix=np.ones((x_matrix.shape[0],x_matrix.shape[1]))
    x_lambda,b=np.linalg.eig(x_matrix)
    print(x_lambda)
    x_lambda=np.diag(x_lambda)
    d_square=np.dot(x_lambda,ones_matrix) -2*x_matrix+ np.dot(ones_matrix,x_lambda)
    return d_square

def centering_matrix(n):
    j_n=np.eye(n,n)-(1/n)*np.ones((n,n))

    return j_n

def p_matrix(d_square):
    j_n=centering_matrix(d_square.shape[0])
    p=np.dot(j_n,d_square)
    p=np.dot(p,j_n)
    p=-1/2*p

    return p

data=np.loadtxt("data.csv",delimiter=",",usecols=range(1,11))
print(data.shape)

#%%
d_square=d_square_matrix_function(data)

p=p_matrix(d_square)

#%%
# 特異値分解は力尽きた
u,s,a= np.linalg.svd(p, full_matrices=True)

plt.scatter(a[:,0],a[:,1])
plt.show()