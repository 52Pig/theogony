

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        # 节点名称
        self.name = nameValue
        # 节点次数
        self.count = numOccur
        # 指向下一个相似节点的指针
        self.nodeLink = None
        # 指向父节点的指针
        self.parent = parentNode
        # 指向子节点的字典
        # key 子节点的元素名称，value 指向子节点的指针
        self.children = {}
        # 节点名称、次数、横向连接、纵向（父亲、孩子节点）
        # 增加节点次数值
    def inc(self, numOccur):
        self.count += numOccur
    # 输出节点和子节点的FP树结构
    # 输出时显得有层次结构，自己输出1个空格
    def disp(self, ind=1):
        print(' ', ind, self.name, ' ', self.count)
        # 依次遍历子节点，输出子节点名称与值
        for child in self.children.values():
            # 子节点空格数加 1
            child.disp(ind + 1)

def ascendTree(leafNode, prefixPath):
    """
    将当前节点leafNode，添加到前缀路径prefixPath的末尾
    然后递归添加其父节点，即往回寻找直到树根节点
    最终prefixPath就是一条从treeNode(包括treeNode)
    到根节点（不包括根节点）的路径。
    """
    # 父节点不为空
    if leafNode.parent != None:
        # 该节点条件到前缀路径中
        prefixPath.append(leafNode.name)
        # 递归调用直到树根节点
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    """
    为给定元素项生成一个条件模式基（前缀路径），
    通过访问树种所有包含给定元素项的节点来完成。
    输入：basePet要挖掘的元素项
    treeNode 为当前FP树种对应的第一个节点。来自于头指针表
    通过 headerTable[basePat][1] 获取
    """
    condPats = {}
    # 如r软属于项根节点，从根节点开始寻找其余相似节点
    while treeNode != None:
        # 存放前缀路径
        prefixPath = []
        # 从该节点往回直到树根节点，寻找前缀路径
        ascendTree(treeNode, prefixPath)
        # 前缀路径元素项大于1，即除去自身项还有其他元素
        if len(prefixPath) > 1:
            # 将路径的计数值赋值给该路径，路径不包含树根节点
            # 取prefixPath[1:]，即treeNode的前缀路径。
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 继续下一个相似元素，按上述过程往回寻找
    # 返回该元素项的所有条件模式基
    return condPats

def createTree(dataSet, minSup=1):
    """
    构建FP-tree树
    输入：数据集、最小支持度
    输出：FP树、头指针表
    """
    # 头指针表
    headerTable = {}
    # 遍历数据集2次
    # 第 1 次遍历，创建头指针表
    for trans in dataSet:
        # 遍历记录中的每一项：
        for item in trans:
            # 每项存入头指针表中，初始为0.依次增加该项在数据集出现的次数
            # dataSet[trans] 使用 1 代替也行
            headerTable[item] = headerTable.get(item, 0) + dataSet[dataSet]
    # 遍历头指针表，移除不满足最小支持度的项
    for k in headerTable.keys():
        # 与最小支持度比较
        if headerTable[k] < minSup:
            # 不满足删除该项
            del(headerTable[k])
    # 对所有项去重后为频繁项集
    freqItemSet = set(headerTable.keys())
    # 若频繁1-项集为空，返回空
    # 即所有项都不是频繁项，无需下一步处理
    if len(freqItemSet) == 0:
        return None, None
    # 在头指针表中，每项增加一个数据项。
    # 存放指向相似元素项指针
    # 保存计数值及指向每种类型第一个元素项指针None
    for k in headerTable:
        # 原基础上增加，存放指向相似元素项指针None
        headerTable[k] = [headerTable[k], None]
        print("headerTable[%s] %s" % (k, headerTable[k]))
        # print(headerTable[k])
    # 设置树根节点，命名为Null Set，出现次数为1
    retTree = treeNode('First', 1, None)

    # 第2次遍历数据集，构建FP-tree
    # tranSet和count表示 一条物品组合和其出现的次数
    for tranSet, count in dataSet.items():
        # 对每项物品，记录其出现次数，用于排序
        # key: 每项物品 value: 出现总次数
        localD = {}
        # 遍历物品项中每件物品
        for item in tranSet:
            # 若该物品在频繁1-项集中
            if item in freqItemSet:
                # 记录物品项的次数
                localD[item] = headerTable[item][0]
        print("zero localD %s" % (localD, len(localD)))
        # 有数据时进行排序
        if len(localD) > 0:
            # 排序 如 薯片：7鸡蛋：7面包：7牛奶：6啤酒：4
            # 对于每一条购买记录，按照上述顺序重新排序。
            # 即频率大小进行排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            print("non zero localD %s" % localD)
            # 利用排好序的记录，更新FP树
            updateTree(orderedItems, retTree, headerTable, count)
    # 返回FP树结构，头指针表
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    """
    输入：排好序的物品项items、构建的FP-tree树inTree
    头指针表headerTable、该条记录的计数值，一般为1
    """
    # 若物品项的第1个物品在FP树结构已存在
    if items[0] in inTree.children:
        # 该元素项的计数值+1
        inTree.children[items[0]].inc(count)
    else:
        # 没有这个元素项时，创建一个新节点treeNode
        # 作为子节点添加到树中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 头指针表更新指向新节点
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 头指针表已有指向元素，即该元素已有相似元素
            # 前一个相似元素项节点的指针指向新节点，调用以下函数执行
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 对剩下元素项迭代调用updateTree函数
    # 不断调用自身，每次调用时会去掉列表中的第一个元素
    # 通过items[1::] 实现
    if len(items) > 1:
        # 记录中还有元素项，递归调用
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    """
    从头指针的nodeToTest开始，一直沿着nodeLink直到到达链表末尾，
    然后将链表末尾指向新节点targetNode 确保节点链接指向树中该元素的每一个实例
    """
    # 找到链表结尾
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    # 链表尾节点指向新节点
    nodeToTest.nodeLink = targetNode


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    输入：inTree为生成的FP树、头指针headerTable
    最小支持度minSup
    preFix空集合set(),保存当前频繁元素项
    freqItemList空列表[],保存生成的频繁项集
    """
    # 对头指针表中的元素按照出现的频率进行从小到大排序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    # 从满足最小支持度的头指针表的最小元素项开始
    for basePat in bigL:
        # 保存当前频繁元素项basePat
        newFreqSet = preFix.copy()
        # 当前频繁元素项basePat加入集合newFreqSet中
        newFreqSet.add(basePat)
        # 将每个频繁项添加到频繁项集列表freqItemList中
        freqItemList.append(newFreqSet)

        # 递归调用findPrefixPath函数，找到元素项basePat的前缀路径
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # 根据当前元素项生成的前缀路径和最小支持度，生成条件树
        myCondTree, myHead = createTree(condPattBases, minSup)
        # 若条件FP树中有元素项，可以再次递归生成条件树
        if myHead != None:
            # 递归挖掘该条件树
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's',],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def loadMovie():
    file = open("ml-100k/u1.base")
    middle = {}
    for line in file.readlines():
        uid, mid, _, _ = line.split("\t")
        if uid not in middle.keys():
            middle[uid] = []
        if len(middle[uid]) < 11:
            middle[uid].append(int(mid))
    return middle.values()

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def fpGrowth(dataSet, minSup=3):
    """
    输入：数据集、最小支持度
    """
    initSet = createInitSet(dataSet)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    freqItems = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), freqItems)
    return freqItems

if __name__ == '__main__':
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()

    # print(findPrefixPath('x', myHeaderTab['x'][1]))
    # print(findPrefixPath('t', myHeaderTab['t'][1]))
    # print(findPrefixPath('z', myHeaderTab['z'][1]))
    # print(myHeaderTab)
    # freqItems = []
    # mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    # print(freqItems)






