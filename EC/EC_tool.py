import random
import numpy as np

"""
    工具类文件，主要有轮盘赌选择算法、计算节点间Jaccard k阶相似度函数、 计算节点间相似度矩阵、计算社区间归一化互信息矩阵、 读取边列表文件并预处理

"""
def roulette(select_arr, seed):
    """
    轮盘赌选择算法

    参数:
    - select_arr: 选择数组
    - seed: 随机种子

    返回:
    - int: 被选中的下标
    """
    np.random.seed(seed)
    sum_val = sum(select_arr)
    if sum_val == 0:
        print()
    random_val = np.random.random()
    probability = 0  # 累计概率
    for i in range(len(select_arr)):
        probability += select_arr[i] / sum_val  # 加上该个体的选中概率
        if probability >= random_val:  # 表示被选中
            return i  # 返回被选中的下标
        else:
            continue


def jaccard_simi(node1, node2, G):
    """
    计算节点之间的 Jaccard 相似度
    """
    return len(G[node1].keys() & G[node2].keys()) / len(G[node1].keys() | G[node2].keys())  # 交集的大小 / 并集的大小


def cal_simi(G, k=1):
    """
    计算节点间的相似度矩阵
    一开始就计算好，避免重复计算。
    参数： G 网络图的邻接表示
    k Jaccard相似度的阶数  默认为1   因为标准的jaccard仅考虑节点直接相连的邻居节点，默认为1   可以考虑不同阶数的相似度
    :param G:{u:{neighbor Vs}}
    :return:simi_dict:{u:{v:simi(u, v)}}
    返回：节点间的相似度矩阵
    """
    simi_dict = {}
    if k == 0:
        return simi_dict
    visit_sequence = sorted(list(G.keys()))  # 得到遍历节点的列表
    # 随机访问
    for u in visit_sequence:
        simi_dict[u] = {}
        for v in visit_sequence:
            if u == v:
                simi_dict[u][v] = 0   # 同一节点相似度设为0
            elif u < v:
                simi_dict[u][v] = jaccard_simi_k(u, v, G, k=k)   # 按序号顺序遍历，
            else:
                simi_dict[u][v] = simi_dict[v][u]   # 保证对称性
    return simi_dict   # 返回相似度矩阵 是对称的


def cal_NMI(comms):
    """
    计算归一化互信息矩阵。 作为社区之间相似度度量

    参数:
    - comms: 社区列表，每个元素是一个社区的节点标识列表

    返回:
    - np.array: 归一化互信息矩阵
    """
    import numpy as np
    res = np.zeros((len(comms), len(comms)))
    # res = [[0]*len(comms)]*len(comms)  这个错误显现了python list是传地址
    # 遍历社区列表每个社区
    for i in range(len(comms)):
        for j in range(i, len(comms)):
            if i == j:
                NMI = 1   # 同一社区归一化互信息为1
            else:
                from sklearn.metrics import normalized_mutual_info_score as nmi
                NMI = nmi(comms[i], comms[j])   # 计算归一化互信息
            res[i][j] = NMI
            res[j][i] = NMI   # 对称性
    mask = np.eye(len(comms), dtype=bool)  # 创建一个布尔单位矩阵 对角线上元素为true
    res[mask] = 0  # 将社区自身相似度置为0
    return res


def jaccard_simi_k(node1, node2, G, k=2):
    '''
    计算k 阶 Jaccard相似度
    G是邻接表表示   {node: {neighbor : weight}
    '''
    # if k==1 :
    # return len(G[node1].keys()&G[node2].keys())/len(G[node1].keys()|G[node2].keys())
    # else:
    node1_neighbors = [node1]       # 将自身添加到邻居列表中
    node2_neighbors = [node2]
    # 遍历K阶邻居
    for i in range(k):
        node1_neighbors_i = []
        node2_neighbors_i = []
        # 获取节点1的k阶邻居
        for neighbor in list(node1_neighbors):
            node1_neighbors_i.extend(list(G[neighbor].keys()))
        node1_neighbors_i = set(node1_neighbors_i)   # 转换为集合类型  去除重复邻居节点
        # 获取节点2的k阶邻居
        for neighbor in list(node2_neighbors):
            node2_neighbors_i.extend(list(G[neighbor].keys()))
        node2_neighbors_i = set(node2_neighbors_i)
        # 更新节点1和节点2的邻居列表为第i阶邻居
        node1_neighbors = node1_neighbors_i.copy()
        node2_neighbors = node2_neighbors_i.copy()

    return len(node1_neighbors & node2_neighbors) / len(node1_neighbors | node2_neighbors)


def read_edgelist(filePath):
    """
    读取边列表文件并修正节点编号从0开始
    返回：list 修正后的边列表，每个元素是包含两个节点标识的列表
    """
    # 修正 从0开始的依次idx
    # 初始化边列表和节点名称集合
    edgelist = []
    names = set()  # name 不一定从0开始以此递增，idx为此
    with open(filePath, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            edgelist.append([int(tmp[0]), int(tmp[1])])
            # 添加节点名称到集合中
            if int(tmp[0]) not in names:
                names.add(int(tmp[0]))
            if int(tmp[1]) not in names:
                names.add(int(tmp[1]))
    # 对节点名称进行排序并构建名称到编号的映射
    nodeId_list = sorted(list(names))
    name2id = {}
    # 判断节点编号是否需要修正
    switch = max(names) == (len(names) + 1)
    # 构建名称到标号的映射
    for idx, name in enumerate(nodeId_list):
        name2id[name] = idx
    # 修正边列表的节点编号
    for edgeIdx in range(len(edgelist)):
        edgelist[edgeIdx][0] = name2id[edgelist[edgeIdx][0]]
        edgelist[edgeIdx][1] = name2id[edgelist[edgeIdx][1]]
    return edgelist



# if __name__ == "__main__":
#     name = "cora"
#     funcName = "dice"
#     ratio = 0.01
#     graphName = "{}-{}-{}-attack".format(name, funcName, ratio)   # 获取图名称和处理方式  类如 cora-dice-0.01-attack
#     print(graphName)
#     name = graphName.split('-')[0]  # 获取图名称
#     print(name)
#     graphPath = "..//network//{}//{}.txt".format(name, graphName)  # 获取图文件路径
#     print(graphPath)
#     edges = read_edgelist(graphPath)    # 读取边列表文件
#     print(edges)
