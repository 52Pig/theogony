import math
from collections import defaultdict

class MaxEnt(object):
    def __init__(self):
        self.feats = defaultdict(int)
        self.trainset = []  # 训练集
        self.labels = set()  # 标签集合

    def load_data(self, file):
        for line in open(file):
            fields = line.strip().split()
            if len(fields) < 2: continue  # 特征数>2列
            label = fields[0]  # 第一列是标签
            self.labels.add(label)
            for f in set(fields[1:]):
                self.feats[(label, f)] += 1  # (label, f) 元组是特征 # print label, f
            self.trainset.append(fields)

    def _initparams(self):  # 初始化参数
        self.size = len(self.trainset)
        self.M = max([len(record)-1 for record in self.trainset])  # GIS训练算法的M参数
        self.ep_ = [0.0] * len(self.feats)
        for i, f in enumerate(self.feats):
            self.ep_[i] = float(self.feats[f]) / float(self.size)  # 计算经验分布的特征期望
            self.feats[f] = i  # 为每个特征函数分配id
        self.w = [0.0] * len(self.feats)  # 初始化权重
        self.lastw = self.w

    def probwgt(self, features, label):  # 计算每个特征权重的指数
        wgt = 0.0
        for f in features:
            if (label, f) in self.feats:
                wgt += self.w[self.feats[(label, f)]]
        return math.exp(wgt)

    '''
    calculate feature expectation on model distribution
    '''
    def Ep(self):  # 特征函数
        ep = [0.0] * len(self.feats)
        for record in self.trainset:  # 从训练集中迭代输出特征
            features = record[1:]
            prob = self.calprob(features)  # 计算条件概率 p(y|x)
            for f in features:
                for w, l in prob:
                    if (l, f) in self.feats:  # 来自训练数据的特征
                        idx = self.feats[(1, f)]  # 获取特征 id
                        ep[idx] += w * (1.0/self.size)  # sum(1/N * f(y, x)*p(y|x),p(x) = 1/N
        return ep

    def _convergence(self, lastw, w):  # 收敛—终止条件
        for w1, w2 in zip(lastw, w):
            if abs(w1 - w2) >= 0.01: return False
        return True

    def train(self, max_iter=1000):  # 训练样本的主函数。默认迭代次数1000次
        self._initparams()  # 初始化参数
        for i in range(max_iter):
            print('iter %d ...' % (i+1))
            self.ep = self.Ep()  # 计算模型分布的特征期望
            self.lastw = self.w[:]
            for i , win in enumerate(self.w):
                delta = 1.0 / self.M * math.log(self.ep_[i] / self.ep[i])
                self.w[i] += delta  # 更新 w
            print(self.w, self.feats)
            if self._convergence(self.lastw, self.w):  # 判断算法是否收敛
                break

    def calprob(self, features):  # 计算条件概率
        wgts = [(self.probwgt(features, l), l) for l in self.labels]  #
        Z = sum([ w for w,l in wgts])  # 归一化参数
        prob = [ (w/Z, l) for w,l in wgts ]  # 概率向量
        return prob

    def predict(self, input):  # 预测函数
        features = input.strip().split()
        prob = self.calprob(features)
        prob.sort(reverse=True)
        return prob


if __name__ == '__main__':
    model = MaxEnt()
    model.load_data('data.txt')  # 导入训练集
    model.train()  # 训练模型
    print(model.predict("Rainy Happy Dry"))  # 预测结果



'''
data.txt:
Outdoor Sunny 5
Outdoor Happy 5
Outdoor Dry 2
Outdoor Humid 6
Outdoor Sad 4
Outdoor Cloudy 4
Indoor Rainy 4
Indoor Humid 4
Indoor Happy 2
Indoor Dry 2
Indoor Sad 4
Indoor Cloudy 2 
'''






