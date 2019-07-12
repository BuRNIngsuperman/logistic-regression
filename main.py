import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from dataloader import get_dataset
from regression import *
from predict import predict


# 构建模型

def model(X_train,Y_train,X_test,Y_test,epoch,learning_rate,print_cost = False):

    # 初始化参数
    w,b = initializa_with_zeros(X_train.shape[0])

    # 梯度下降
    parameters,grads,costs = optimize(w,b,X_train,Y_train,epoch,learning_rate,print_cost)

    # 获取参数w,b
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)


    # 输出 train/test 误差

    print("train accurancy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accurancy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test))*100))

    #输出字典

    d = {"costs":costs,
         "Y_prediction_train":Y_prediction_train,
         "Y_prediction_test":Y_prediction_test,
         "w":w,
         "b":b,
         "epoch":epoch,
         "learning_rate":learning_rate}

    return d


if __name__ == "__main__":
    train_x_orig, train_y, test_x_orig, test_y, classes = get_dataset()

    # Reshape train/test samples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # # Example of a picture
    # index = 2
    # plt.imshow(train_x_orig[index])
    # plt.show()
    # print("y = " + str(train_y[:, index]) + ", it's a '" + classes[np.squeeze(train_y[:, index])].decode(
    #     "utf-8") + "' picture.")
    # print(train_x_orig.shape)
    # print(train_x_flatten.shape)
    # 对图片数据集进行标准化处理
    train_x = train_x_flatten/255
    test_x = test_x_flatten/255
    d = model(train_x, train_y, test_x, test_y, epoch=2000, learning_rate=0.005, print_cost=True)