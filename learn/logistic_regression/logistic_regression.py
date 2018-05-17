import numpy as np
import time
import random

# 优点：
# 实现简单、易于理解实现、计算代价不高、速度快、存储低
# 缺点：
# 容易欠拟合、分类器精度不高
# 应用：ctr

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    # 步长
    alpha = 0.01
    # 最大迭代次数
    maxCycles = 500
    # 权重矩阵
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转为数组，并返回
    return weights.getA()

def colicTest():
    frTrain = open('../data/horseColicTraining.txt')
    frTest = open('../data/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(curLine) - 1):
            lineArr.append(float(curLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(curLine[-1]))
    # 使用梯度上升求解
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(curLine) - 1):
            lineArr.append(float(curLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:,0])) != int(curLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100
    print("测试集错误率为: %.2f%%" % errorRate)

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

if __name__ == '__main__':
    start = time.time()
    colicTest()
    print("cost time{}".format(time.time()-start))











