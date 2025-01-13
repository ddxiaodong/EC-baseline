# coding=utf-8
import collections
import string
import random
from tqdm import tqdm
'''
    paper : <<Fast unfolding of communities in large networks>>
'''

def load_graph(path):
    G = collections.defaultdict(dict)
    with open(path) as text:
        for line in text:
            vertices = line.strip().split()
            v_i = int(vertices[0])
            v_j = int(vertices[1])
            G[v_i][v_j] = 1.0
            G[v_j][v_i] = 1.0
    return G

# 节点类 存储社区与节点编号信息
class Vertex:
    def __init__(self, vid, cid, nodes, k_in=0):
        # 节点编号
        self._vid = vid
        # 社区编号
        self._cid = cid
        self._nodes = nodes
        self._kin = k_in  # 结点内部的边的权重


class Louvain:
    def __init__(self, G, clearN=None, seed=0):
        self._G = G
        self._m = 0  # 边数量 图会凝聚动态变化
        self._cid_vertices = {}  # 需维护的关于社区的信息(社区编号,其中包含的结点编号的集合)
        self._vid_vertex = {}  # 需维护的关于结点的信息(结点编号，相应的Vertex实例)
        for vid in self._G.keys():
            # 刚开始每个点作为一个社区
            self._cid_vertices[vid] = {vid}
            # 刚开始社区编号就是节点编号
            self._vid_vertex[vid] = Vertex(vid, vid, {vid})
            # 计算边数  每两个点维护一条边
            self._m += sum([1 for neighbor in self._G[vid].keys()
                            if neighbor > vid])
        self.numberOfnodes = len(G) if clearN==None else clearN
        self.seed = seed
    # 模块度优化阶段

    def first_stage(self):
        mod_inc = False  # 用于判断算法是否可终止
        visit_sequence = self._G.keys()
        # 随机访问
        random.seed = self.seed
        random.shuffle(list(visit_sequence))
        while True:
            can_stop = True  # 第一阶段是否可终止
            # 遍历所有节点
            for v_vid in visit_sequence:
            # for idx in tqdm(range(len(visit_sequence))):
            #     v_vid = visit_sequence[idx]
                #print(v_vid)
                # 获得节点的社区编号
                v_cid = self._vid_vertex[v_vid]._cid
                # k_v节点的权重(度数)  内部与外部边权重之和
                k_v = sum(self._G[v_vid].values()) + \
                      self._vid_vertex[v_vid]._kin
                # 存储模块度增益大于0的社区编号
                cid_Q = {}
                # 遍历节点的邻居
                for w_vid in self._G[v_vid].keys():
                    # 获得该邻居的社区编号
                    w_cid = self._vid_vertex[w_vid]._cid
                    if w_cid in cid_Q:
                        continue
                    else:
                        # tot是关联到社区C中的节点的链路上的权重的总和
                        tot = sum(
                            [sum(self._G[k].values()) + self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
                        if w_cid == v_cid:
                            tot -= k_v
                        # k_v_in是从节点i连接到C中的节点的链路的总和
                        k_v_in = sum(
                            [v for k, v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
                        # 由于只需要知道delta_Q的正负，所以少乘了1/(2*self._m)
                        delta_Q = k_v_in - k_v * tot / self._m
                        cid_Q[w_cid] = delta_Q

                # 取得最大增益的编号
                cid, max_delta_Q = sorted(
                    cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
                # 在这补充相似度进行轮盘赌，既考虑模块度增益，又考虑相似度和随机性
                if max_delta_Q > 0.0 and cid != v_cid:
                    # 让该节点的社区编号变为取得最大增益邻居节点的编号
                    #print("点id：{};原社团：{}；新社团：{}".format(v_vid, v_cid, cid))
                    self._vid_vertex[v_vid]._cid = cid
                    # 在该社区编号下添加该节点
                    self._cid_vertices[cid].add(v_vid)
                    # 以前的社区中去除该节点
                    self._cid_vertices[v_cid].remove(v_vid)
                    # 模块度还能增加 继续迭代
                    can_stop = False
                    mod_inc = True

            if can_stop:
                break
        return mod_inc

    # 网络凝聚阶段
    def second_stage(self):
        cid_vertices = {}
        vid_vertex = {}
        # 遍历社区和社区内的节点
        for cid, vertices in self._cid_vertices.items():
            if len(vertices) == 0:
                continue
            new_vertex = Vertex(cid, cid, set())
            # 将该社区内的所有点看做一个点
            for vid in vertices:
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                # k,v为邻居和它们之间边的权重 计算kin社区内部总权重 这里遍历vid的每一个在社区内的邻居   因为边被两点共享后面还会计算  所以权重/2
                for k, v in self._G[vid].items():
                    if k in vertices:
                        new_vertex._kin += v / 2.0
            # 新的社区与节点编号
            cid_vertices[cid] = {cid}
            vid_vertex[cid] = new_vertex

        G = collections.defaultdict(dict)
        # 遍历现在不为空的社区编号 求社区之间边的权重
        for cid1, vertices1 in self._cid_vertices.items():
            if len(vertices1) == 0:
                continue
            for cid2, vertices2 in self._cid_vertices.items():
                # 找到cid后另一个不为空的社区
                if cid2 <= cid1 or len(vertices2) == 0:
                    continue
                edge_weight = 0.0
                # 遍历 cid1社区中的点
                for vid in vertices1:
                    # 遍历该点在社区2的邻居已经之间边的权重(即两个社区之间边的总权重  将多条边看做一条边)
                    for k, v in self._G[vid].items():
                        if k in vertices2:
                            edge_weight += v
                if edge_weight != 0:
                    G[cid1][cid2] = edge_weight
                    G[cid2][cid1] = edge_weight
        # 更新社区和点 每个社区看做一个点
        self._cid_vertices = cid_vertices
        self._vid_vertex = vid_vertex
        self._G = G

    def get_communities(self):
        communities = []
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(list(c))
        return communities
    def get_labels(self):
        label = [0]*self.numberOfnodes
        cid = -1
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                cid += 1
                for vid in vertices:
                    for idx in self._vid_vertex[vid]._nodes:
                        label[idx] = cid
        return label
    def execute(self, out_label=True):
        iter_time = 1
        while True:
            #print(iter_time)
            iter_time += 1
            # 反复迭代，直到网络中任何节点的移动都不能再改善总的 modularity 值为止
            #print("first start")
            mod_inc = self.first_stage()
            #print("first end")
            if mod_inc:
                self.second_stage()
            else:
                break
        if out_label:
            return self.get_labels()
        else:
            return self.get_communities()


def evaluate(true_label, label):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    return nmi(true_label, label)
def main_qattack():
    name = 'dblp'
    lamb = 1
    G = load_graph('../network/{}-qattack.txt'.format(name))

    import pickle
    label_path = "..//data//{}.label".format(name)
    with open(label_path, 'rb') as f:
        true_label = pickle.load(f)
    commNum = max(true_label)+1


    clearN = len(true_label)

    if clearN==len(G):
        algorithm = Louvain(G)
    else:
        algorithm = Louvain(G, clearN)
    labels = algorithm.execute()
    if clearN!=len(G):
        delErrNode_evaluate(true_label, labels, G)
    else:
        NMI = evaluate(true_label, labels)
        print(NMI)
def main_random():
    name = "email-Eu-core"
    ratio = 0
    part = 0
    G = load_graph("..//network//{}//{}-ratio_{}-part{}.txt".format(name,name, ratio, part))
    algorithm = Louvain(G)
    labels = algorithm.execute()
    import pickle
    label_path = "..//data//{}.label".format(name)
    with open(label_path, 'rb') as f:
        true_label = pickle.load(f)
    NMI = evaluate(true_label, labels)
    print(NMI)
def main_attack(name="citeseer", funcName="dice", ratio=0.01, seed = 0):
    # name = "football"
    # funcName = "dice"
    # ratio = 0.01
    graphPath ="..//network//{}//{}-{}-{}-attack.txt".format(name, name, funcName, ratio)
    labelPath = '..//network//{}//{}.label'.format(name, name)
    import pickle

    with open(labelPath, 'rb') as f:
        true_label = pickle.load(f)
    number_of_class = max(true_label)
    clearN = len(true_label)
    G = load_graph(graphPath)
    if clearN==len(G):
        algorithm = Louvain(G, seed=seed)
    else:
        algorithm = Louvain(G, clearN, seed=seed)
    labels = algorithm.execute()
    if clearN!=len(G):
        NMI = delErrNode_evaluate(true_label, labels, G)
    else:
        NMI = evaluate(true_label, labels)
        print(NMI)

    return NMI
def delErrNode_evaluate(true_label, labels, G):
    """被攻击后的图会出现孤立的点，在计算NMI时需要去掉"""
    """得到哪些点是孤立的"""
    del_indexs = set([i for i in range(len(true_label))])
    save_indexs = list(del_indexs & set(G.keys()))
    del_true_label = [true_label[idx] for idx in save_indexs]
    del_label = [labels[idx] for idx in save_indexs]
    NMI = evaluate(del_true_label, del_label)
    print(NMI)
    return NMI
def xunhuan():
    import pickle
    path = "cora-ec1.G"
    with open(path, 'rb') as f:
        G = pickle.load(f)
    m = 0
    m_new = 0
    new_G = {}
    for key, value in G.items():
        m += len(G[key])
        new_G[key]={}
        for v, w in value.items():
            if w>=1:
                new_G[key][v] = w
        m_new += len(new_G[key])

    print(m, m_new)
    algorithm = Louvain(new_G)
    labels = algorithm.execute()
    #print(labels)
    import pickle
    name = 'cora'
    label_path = "..//data//{}.label".format(name)
    with open(label_path, 'rb') as f:
        true_label = pickle.load(f)
    delErrNode_evaluate(true_label=true_label, labels=labels, G=new_G)
    # NMI = evaluate(true_label, labels)
    # print(NMI)

def experiment():
    import numpy as np
    name = 'cora'
    funcName = 'dice'
    ratio = 0.01
    fileName = '{}-{}-{}-attack'.format(name, funcName, ratio)





    time = 10
    resList = {0:[], 1:[], 2:[]}  # 0:NMI1, 1:NMI2, 2:NMI3
    for seed in range(time):

        seedRes = main_attack(name, funcName, ratio, seed=seed)
        #seedRes =main_qattack(name=name, lamb=lamb, k=k, classNum=classNum, j_k=j_k, seed=seed)
        resList[0].append(seedRes)



    NMI1_mean = np.array(resList[0]).mean()
    NMI1_std = np.array(resList[0]).std()


    from decimal import Decimal
    NMI1Str = "{}±{}".format(Decimal(NMI1_mean).quantize(Decimal("0.0001")), Decimal(NMI1_std).quantize(Decimal("0.0001")))


    path = "..//output_lou//"
    resName = "{}-{}-{}-attack-result-lou.txt".format(name, funcName, ratio,)
    with open(path+resName, 'w') as f:
        resClu = "改进结果：{}\n".format(NMI1Str)
        f.writelines(resClu)
        for seed in range(time):
            resStr = "{}, {}\n".format(seed, resList[0][seed])
            f.writelines(resStr)
if __name__ == '__main__':
    import time

    time_start=time.time()

    experiment()
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
        # for vid in self._G.keys():
        #     self._m += sum([self._G[vid][neighbor] for neighbor in self._G[vid].keys() if neighbor>vid])