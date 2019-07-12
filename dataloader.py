import numpy as np
import h5py

# 读取数据
def get_dataset():
    train_dataset = h5py.File("datasets/train_catvnoncat.h5","r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File("datasets/test_catvnoncat.h5","r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_y_orig = train_y_orig.reshape(1,train_y_orig.shape[0])
    test_y_orig = test_y_orig.reshape(1,test_y_orig.shape[0])

    return  train_x_orig,train_y_orig,test_x_orig,test_y_orig,classes


if __name__ == "__main__":
    x_train,y_train,x_test,y_test,classes = get_dataset()
    print("训练集输入x的维度是" +str(x_train.shape))
    print("训练集输出y的维度是" +str(y_train.shape))
    print("测试集输入x的维度是" +str(x_test.shape))
    print("测试集输出y的维度是" +str(y_test.shape))
