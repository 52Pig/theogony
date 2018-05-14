
import math
from scipy.spatial import distance
import numpy as np

class UserBasedCF:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.readData()

    def readData(self):
        self.train = dict()
        for line in open(self.train_file):
            user, item, score, _ = line.strip().split("\t")
            self.train.setdefault(user, {})
            self.train[user][item] = int(score)

        self.test = dict()
        for line in open(self.test_file):
            user, item, score, _ = line.strip().split("\t")
            self.test.setdefault(user, {})
            self.test[user][item] = int(score)
    def UserSimilarity(self):
        self.item_users = dict()
        for user, items in self.train.items():
            for i in items.keys():
                if i not in self.item_users:  # 如果是第一个商品
                    self.item_users[i] = set()
                self.item_users[i].add(user)
        C = dict()
        N = dict()
        Cor = dict()
        for i, users in self.item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    if u == v:  #  自己和自己是最相关的，需排除掉
                        continue
                    C[u].setdefault(v, 0)
                    C[u][v] += 1
                Cor.setdefault(u, [])
                for v in users:
                    if u == v:
                        continue
                    Cor[u].append(v)
        self.W = dict()
        self.Euc = dict()
        for u, related_users in C.items():
            self.W.setdefault(u, {})
            for v, cuv in related_users.items():
                self.W[u][v] = cuv / math.sqrt(N[u] * N[v])
                # self.Euc[u][v] = distance.cdist(N[u], N[v], 'euclidean')
                self.Euc[u][v] = np.sum(distance.cdist([Cor[u][:10]], [Cor[v][:10]], 'euclidean'))  # 欧式解决
                # self.Cos[u][v] = np.sum(distance.cdist([Cor[u][:10]], [Cor[v][:10]], 'cosine'))  # 余弦距离
                # self.Man[u][v] = np.sum(distance.cdist([Cor[u][:10]], [Cor[v][:10]], ''))  # 曼哈顿距离
                # print(self.W[u][v])
                # print(self.Euc[u][v])
                pass
        return self.W

    def Recommend(self, user, K=3, N=10):
        rank = dict()
        action_item = self.train[user].keys()

        for v, wuv in sorted(self.W[user].items(), key=lambda x:x[1], reverse=True)[0:K]:
            for i, rvi in self.train[v].items():
                if i in action_item:
                    continue
                rank.setdefault(i, 0)
                rank[i] += wuv * rvi
        return sorted(rank.items(), key=lambda x:x[1], reverse=True)[0:N]

if __name__ == '__main__':
    cf = UserBasedCF('u.data', 'u.data')
    cf.UserSimilarity()
    print(cf.Recommend("a"))













