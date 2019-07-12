import numpy as np
import scipy
from scipy import ndimage

# activate function
def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return  z

# initializa_parameters

def initializa_with_zeros(dim):

    w = np.zeros((dim,1))
    b=0

    assert(w.shape == (dim,1))
    assert (isinstance(b,int) or isinstance(b,float))

    return w,b

# 传播算法
def propagate(w,b,X,Y):
    m = X.shape[1]

    # 前向传播
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # 反向传播
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw":dw,
             "db":db}

    return grads,cost

# 梯度下降算法
def optimize(w,b,X,Y,epoch,learning_rate,print_cost = False):

    costs = []

    for i in  range(epoch):

        # 获取梯度和代价函数
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        # 更新权重

        w = w - learning_rate*dw
        b = b - learning_rate*db

        # 记录cost值
        if i%100 ==0:
            costs.append(cost)

        if print_cost == 0 and i%100 == 0 :
            print("cost after epoch %d : %f" %(i,cost))

    params = {"w":w,
            "b":b}

    grads = {"dw": dw,
            "db": db}

    return params, grads,costs

