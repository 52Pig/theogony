# -*- coding:utf-8 -*-

from numpy import *

startP = mat([0.63, 0.17, 0.20])  # 起始概率
stateP = mat([[0.5, 0.25, 0.25], [0.375, 0.125, 0.375], [0.125, 0.675, 0.375]])
# 状态转移概率
emitP = mat([[0.6, 0.20, 0.05], [0.25, 0.25, 0.25], [0.05, 0.10, 0.50]])  # 发射(混合)概率
state1Emit = multiply(startP, emitP[:, 0].T)  # 计算概率：干旱-干燥-潮湿
print(state1Emit)
print('argmax', state1Emit.argmax())
# 计算干燥的概率
state2Emit = stateP * state1Emit.T
state2Emit = multiply(state2Emit, emitP[:, 1])
print(state2Emit.T)
print('argmax:', state2Emit.argmax())
# 计算潮湿的概率
state3Emit = stateP * state2Emit
state3Emit = multiply(state3Emit, emitP[:, 2])
print(state3Emit.T)
print('argmax:', state3Emit.argmax())

