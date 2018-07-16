# -*- coding: UTF-8 -*-
# import sys
import numpy as np
#from importlib import reload
from ilearn.lstm.lstm import LstmParam, LstmNetwork, LossLayer

# reload(sys)
# sys.setdefaultencoding('utf-8')

np.random.seed(0)


def gensamples(x_dim):
    ylabels = [-0.5, 0.2, 0.1, -0.5]
    xinputs = [np.random.random(x_dim) for i in ylabels]  # 对应输出矩阵的一系列随机数
    return xinputs, ylabels


if __name__ == "__main__":
    x_dim = 50  # 输出维度
    maxiter = 100  # 最大迭代次数
    # input_val -> X: 50 维的随机数向量; y_list->y: 每个Xi向量对应的一个y的输出值,一个四列
    # Xi[0:50] -> yi
    input_val_arr, y_list = gensamples(x_dim)

    # 初始化 lstm 各部分参数
    mem_cell_ct = 100  # 存储单元维度
    concat_len = x_dim + mem_cell_ct  # 输入维度与存储单元维度之和

    lstm_param = LstmParam(mem_cell_ct, x_dim)  # 初始化 lstm 神经网络的参数
    lstm_net = LstmNetwork(lstm_param)  # 创建 lstm 神经网络对象
    for cur_iter in range(maxiter):
        ypredlist = []
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            ypredlist.append((ind, lstm_net.lstm_node_list[ind].state.h[0]))

        loss = lstm_net.y_list_is(y_list, LossLayer)  # 计算全局损失
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

        if (cur_iter+1)%10==0:
            print("cur iter: ", cur_iter)
            print("y_pred: ", ypredlist)
            print("loss: ", loss)
