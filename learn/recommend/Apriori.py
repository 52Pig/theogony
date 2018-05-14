from numpy import *

def loadDataSet():
    return [[1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]

def loadUseful():
    file = open("abc.txt")
    middle = {}
    ret = []
    for line in file.readlines():
        uid, mid, _, _ = line.split("\t")
        if uid not in middle.keys():
            middle[uid] = []
        middle[uid].append(int(mid))
    for k, v in middle.items():
        ret.append(v)
    return ret

def createC1(dataSet):
    """
    遍历数据集，建立频繁1项集
    """
    C1 = []
    # 遍历每条记录
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:  # 该物品没在列表中
                C1.append([item])
    C1.sort()  # 对所有物品排序
    return map(frozenset, C1)

def scanD(D, Ck, minSupport):
    """
    建立字典<key,value>
    候选集Ck中每项及在所有物品记录中出现的次数
    key -- > 候选集中的每项
    value -- > 该物品在所有物品记录中出现的次数
    """
    ssCnt = {}
    # 对比候选集中的每项与原物品记录，统计出现次数
    for tid in D:
        for can in Ck:
            # 如果候选集Ck中该项在该条物品记录出现
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] += 1
                else:
                    ssCnt[can] += 1
    # 数据集中总的记录数，物品购买记录总数，用于计算支持度
    numItems = float(len(D))
    # 记录经最小支持度过滤后的频繁项集
    retList = []
    # key：候选集中满足条件的项；value：该项支持度
    supportData = {}
    for key in ssCnt:
        # 计算每项的支持度
        support = ssCnt[key]/numItems
        if support >= minSupport:
            # 只是为了让列表看起来有组织
            retList.insert(0, key)
            # 记录该项的支持度
            # 注意: 候选集中所有项的支持度均被保存下来了
            # 不仅仅是满足最小支持度的项，其他项也被保存
        supportData[key] = support
    # 返回满足条件的物品项，以及每项的支持度
    return retList, supportData

def aprioriGen(Lk, k):
    """
    由上层频繁k-1项集生成候选k项集
    如 输入为{0}, {1}, {2}会生成{0, 1}, {0, 2}, {1, 2}
    输入： 频繁k-1项集，新的候选集元素个数k
    输出：候选集
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        # 遍历候选集中除前项后的其他项，与当前项比较
        for j in range(i+1, lenLk):
            # print(list(Lk[i]))
            # 避免列表为负数
            L1 = list(Lk[i])[:k-2]
            # print(L1)
            L2 = list(Lk[j])[:k-2]
            # print(L2)
            # 排序
            # print(L1.sort())
            # print(L2.sort())
            # 相同则两项合并
            if L1==L2:
                # print(Lk[i], Lk[j])
                retList.append(Lk[i] | Lk[j])
            # else:
            #     print(retList)
    return retList

def generateRules(L, supportData, minConf=0.7):
    """
    输入：apriori函数生成频繁项集列表L
    支持度列表、最小置信度
    输出：包含可信度规则的列表
    作用：产生关联规则
    """
    # 置信度规则列表
    bigRuleList = []
    # L0位频繁1-项集
    # 无法从1-项集中构建关联规则，所以从2-项集开始。
    # 遍历L中的每一个频繁项集
    for i in range(1, len(L)):
        # 遍历频繁项集的每一项
        for freqSet in L[i]:
            # 对每个频繁项集构建只包含单个元素集合的列表H1
            # 如 {1,2,3,4}, H1为[{1},{2}, {3}, {4}]
            # 关联规则从单个项开始逐步增加
            # 1,2,3-->4  1,2-->3,4  1-->2,3,4
            H1 = [frozenset([item]) for item in freqSet]
            # 频繁项集中元素大于3个及以上，
            # 规则右部需要不断合并作为整体，利用最小置信度进行过滤
            if (i > 1):
                # 项集中元素超过2个，做合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 满足最小置信度要求的规则列表选项
    prunedH = []
    # 遍历H中的所有项，用作关联规则的后项
    for conseq in H:
        # 置信度计算，使用集合减操作
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        # 置信度大于最小置信度
        if conf >= minConf:
            # 输出关联规则前件freqSet-conseq
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            # 保存满足条件的关联规则
            # 保存关联规则前件，后件，置信度
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    # 返回满足条件的后项
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    H 关联规则右部的元素，如1,2,-->3,4
    频繁项集为{1,2,3,4}，H为3,4
    因此，先计算H大小m(此处m=2)
    """
    # 规则右边的元素个数
    m = len(H[0])
    # {1,2,3}产生规则1-->2,3
    # 规则右边的元素H 最多 比频繁项集freqSet元素少1，
    # 超过该条件无法产生关联规则
    # 如果H元素较少，那么可以对H元素进行组合
    # 产生规则右边新的组合H，
    # 直到达到H元素 最多
    # 若{1,2,3,4},m=2时。可产生如下规则：
    # 1,2-->3,4
    # 1,3-->2,4
    # 1,4-->2,3
    # 2,3-->1,4
    # 2,4-->1,3
    # 3,4-->1,2
    if (len(freqSet) > (m + 1)):
        # 使用apriorGen()函数对H元素进行无重复组合
        # 用于生产更多的候选规则，结果存储在Hmp1中。
        # Hmp1=[[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
        Hmp1 = aprioriGen(H, m+1)
        # 利用最小置信度对这些候选规则进行过滤
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # 过滤后Hmp1=[[1,2,3], [1,2,4]]
        # 如果不止一条规则满足要求
        # 继续使用Hmp1调用函数rulesFromConseq()
        # 判断是否可以进一步组合这些规则
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def apriori(dataSet, minSupport=0.5):
    # 生成频繁1-项集
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    # 过滤最小支持度，得到频繁1-项集以及每项的支持度
    L1, supportData = scanD(D, C1, minSupport)
    for ll in L1:
        print(list(ll))
    # 将L1放入列表L中，L会包含L1，L2，L3
    L = [L1]
    # py中使用下标0作为第一个元素，k=2表示从1-项集产生2-项候选集
    # L0位频繁1-项集
    k = 2
    # 根据L1寻找L2、L3
    # 创建包含更大项集的更大列表，直到下一个更大的项集为空，
    # 候选集物品组合长度超过原数据集最大的物品记录长度
    # 如 原始数据集物品记录最大长度为4，那么候选集最多为4-项集
    while (len(L[k-2]) > 0):
        # 由频繁k-1项集，产生k项候选集
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData



if __name__ == "__main__":
    dataSet = loadDataSet()
    # dataSet = loadUseful()
    L, supportData = apriori(dataSet, minSupport=0.001)
    # rules = generateRules(L, supportData, minConf=0.5)
    # print(L, rules)
    file = open("./apriori", "w")
    for l in L:
        li = []
        for ll in l:
            for k in ll:
                li.append(str(k))
        file.write("&&".join(li) + "\r\n")






