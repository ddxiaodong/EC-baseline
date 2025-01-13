# coding=utf-8
import collections
import string
import random

'''
    paper : <<Fast unfolding of communities in large networks>>
    改进的Louvain算法
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


def load_graph_from_numpy(path):
    """从numpy文件加载邻接矩阵并转换为图字典格式
    
    参数:
    - path: .npy文件路径
    
    返回:
    - G: 图字典 {node_id: {neighbor_id: weight}}
    """
    import numpy as np
    adj_matrix = np.load(path)
    G = collections.defaultdict(dict)
    
    # 将邻接矩阵转换为字典格式
    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adj_matrix[i,j] != 0:
                G[i][j] = float(adj_matrix[i,j])
                
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


class mod_Louvain:
    """
    Louvain算法模型
    G  图
    lamb λ参数
    clearN  真实标签的数量
    simi 相似度矩阵
    """
    def __init__(self, G, lamb=1, clearN=None, simi=None):
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

        from EC_tool import cal_simi
        self._simi = cal_simi(self._G) if simi == None else simi    # 计算节点间的相似度
        # self._orgG = G.copy()
        self.lamb = lamb
        self.numberOfnodes = len(G) if clearN == None else clearN
        # print()

    # 模块度优化阶段
    def first_stage(self, sseed):
        mod_inc = False  # 用于判断算法是否可终止
        visit_sequence = self._G.keys()
        # 随机访问
        import numpy as np
        np.random.seed(sseed)
        np.random.shuffle(list(visit_sequence))
        while True:
            can_stop = True  # 第一阶段是否可终止
            # 遍历所有节点
            for v_vid in visit_sequence:
                # 获得节点的社区编号
                v_cid = self._vid_vertex[v_vid]._cid
                # k_v节点的权重(度数)  内部与外部边权重之和
                k_v = sum(self._G[v_vid].values()) + \
                      self._vid_vertex[v_vid]._kin
                # 存储模块度增益大于0的社区编号
                cid_Q = {}
                cid_simi = {}
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

                        # 这里开始计算相似度
                        # self._vid_vertex[v_vid]._nodes得到本节点（社团）的内部节点->self._orgG[nodeId]得到邻居
                        # self._vid_vertex[w_vid]._nodes得到对应的此时的邻居社团，两者结合计算相似度
                        # cid_simi={w_cid: simi(v_cid, w_cid)}
                        cid_simi[w_cid] = 0
                        for u in self._vid_vertex[v_vid]._nodes:
                            for v in self._vid_vertex[w_vid]._nodes:
                                cid_simi[w_cid] += self._simi[u][v]
                # 取得最大增益的编号
                # cid, max_delta_Q = sorted(
                #     cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
                # cid = sorted(
                #     cid_Q.items(), key=lambda item: item[1], reverse=True)
                # 在这补充相似度进行轮盘赌，既考虑模块度增益，又考虑相似度和随机性
                # 不需要排序了：直接遍历cid_Q.items()
                # Q>0:整合成table_Q_simi = list[[id, Q, simi, P=0], ]
                # if len(table_Q_simi)==0: 不执行； elif == 1: 直接选择cid； elif>2:继续后续操作
                # 归一化后计算P
                # 轮盘赌得到 cid
                table_Q_simi = [[], [], []]  # [[id...], [Q...], [simi...], [P...]]
                for node, Q_v in cid_Q.items():
                    if node == v_cid:
                        continue
                    if Q_v > 0:
                        table_Q_simi[0].append(node)
                        table_Q_simi[1].append(Q_v)
                        table_Q_simi[2].append(cid_simi[node])
                        # table_Q_simi[3].append(0)

                if len(table_Q_simi[0]) != 0:
                    if len(table_Q_simi[0]) == 1:
                        cid = table_Q_simi[0][0]
                    else:
                        # 通过函数计算P，然后轮盘赌得到cid
                        import numpy as np
                        table_Q_simi[1] = np.array(table_Q_simi[1]).reshape(-1, 1)
                        table_Q_simi[2] = np.array(table_Q_simi[2]).reshape(-1, 1)
                        from sklearn import preprocessing
                        min_max_scaler = preprocessing.MaxAbsScaler()
                        # 用min-max归一化
                        tmp1 = min_max_scaler.fit_transform(table_Q_simi[1])
                        tmp2 = min_max_scaler.fit_transform(table_Q_simi[2])
                        # table_Q_simi[1] = min_max_scaler.fit_transform(table_Q_simi[1])
                        # table_Q_simi[2] = min_max_scaler.fit_transform(table_Q_simi[2])

                        # tmp = (table_Q_simi[1]+self.lamb*table_Q_simi[2]).reshape(1,-1).tolist()[0]
                        tmp = (1 * tmp1 + self.lamb * tmp2).reshape(1, -1).tolist()[0]
                        eps = 10 ** (-8)
                        tmp = np.clip(tmp, eps, np.inf)
                        if sum(tmp) == 0:
                            print("error here!")
                            print(table_Q_simi[1])
                            print(table_Q_simi[2])
                        from EC_tool import roulette
                        idx = roulette(tmp, sseed)

                        cid = table_Q_simi[0][idx]
                        # print("点id：{};原社团：{}；新社团：{}".format(v_vid, v_cid, cid))
                    if cid != v_cid:
                        self._vid_vertex[v_vid]._cid = cid
                        # 在该社区编号下添加该节点
                        self._cid_vertices[cid].add(v_vid)
                        # 以前的社区中去除该节点
                        self._cid_vertices[v_cid].remove(v_vid)
                        # 模块度还能增加 继续迭代
                        # can_stop = False  # 由于不是每次都局部最大 会产生多跳循环导致死循环
                        mod_inc = True

                # if max_delta_Q > 0.0 and cid != v_cid:
                #     # 让该节点的社区编号变为取得最大增益邻居节点的编号
                #     self._vid_vertex[v_vid]._cid = cid
                #     # 在该社区编号下添加该节点
                #     self._cid_vertices[cid].add(v_vid)
                #     # 以前的社区中去除该节点
                #     self._cid_vertices[v_cid].remove(v_vid)
                #     # 模块度还能增加 继续迭代
                #     can_stop = False
                #     mod_inc = True
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
        self._cid_vertices = cid_vertices  # 相当于只记录了nodeId
        self._vid_vertex = vid_vertex  # 每个commId对应了原本的哪些nodes
        self._G = G  # 记录nodes(comms)之间的链接（邻居）

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
        label = [0] * self.numberOfnodes
        cid = -1
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                cid += 1
                for vid in vertices:
                    for idx in self._vid_vertex[vid]._nodes:
                        label[idx] = cid
        return label

    def execute(self, seed=0):
        """
        迭代执行Louvain算法的两个阶段，直到网络中任何节点移动不能改善总模块度为止
        返回最终得到的社区标签
        """
        # print("execute!!!!!!!!!!!!")
        iter_time = 1
        while True:
            iter_time += 1
            # 反复迭代，直到网络中任何节点的移动都不能再改善总的 modularity 值为止
            mod_inc = self.first_stage(seed)
            if mod_inc:
                self.second_stage()
            else:
                break
        return self.get_labels()


class mod_Louvain2:
    # def __init__(self, G, lamb=1,clearN=None, simi=None):
    def __init__(self, G, clearN=None, ):
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

        from tool import cal_simi
        # self._simi = cal_simi(self._G) if simi==None else simi
        # self._orgG = G.copy()
        # self.lamb = lamb
        self.numberOfnodes = len(G) if clearN == None else clearN
        # print()

    # 模块度优化阶段
    def first_stage(self, seed):
        import numpy as np
        mod_inc = False  # 用于判断算法是否可终止
        visit_sequence = self._G.keys()
        # 随机访问
        np.random.seed(seed)
        np.random.shuffle(list(visit_sequence))
        while True:
            can_stop = True  # 第一阶段是否可终止
            # 遍历所有节点
            for v_vid in visit_sequence:
                # 获得节点的社区编号
                v_cid = self._vid_vertex[v_vid]._cid
                # k_v节点的权重(度数)  内部与外部边权重之和
                k_v = sum(self._G[v_vid].values()) + \
                      self._vid_vertex[v_vid]._kin
                # 存储模块度增益大于0的社区编号
                cid_Q = {}
                # cid_simi={}
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

                        # 这里开始计算相似度
                        # self._vid_vertex[v_vid]._nodes得到本节点（社团）的内部节点->self._orgG[nodeId]得到邻居
                        # self._vid_vertex[w_vid]._nodes得到对应的此时的邻居社团，两者结合计算相似度
                        # cid_simi={w_cid: simi(v_cid, w_cid)}
                        # cid_simi[w_cid] = 0
                        # for u in self._vid_vertex[v_vid]._nodes:
                        #     for v in self._vid_vertex[w_vid]._nodes:
                        #         cid_simi[w_cid]+=self._simi[u][v]
                # 取得最大增益的编号
                # cid, max_delta_Q = sorted(
                #     cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
                # cid = sorted(
                #     cid_Q.items(), key=lambda item: item[1], reverse=True)
                # 在这补充相似度进行轮盘赌，既考虑模块度增益，又考虑相似度和随机性
                # 不需要排序了：直接遍历cid_Q.items()
                # Q>0:整合成table_Q_simi = list[[id, Q, simi, P=0], ]
                # if len(table_Q_simi)==0: 不执行； elif == 1: 直接选择cid； elif>2:继续后续操作
                # 归一化后计算P
                # 轮盘赌得到 cid
                table_Q_simi = [[], [], []]  # [[id...], [Q...], [simi...], [P...]]
                for node, Q_v in cid_Q.items():
                    if node == v_cid:
                        continue
                    if Q_v > 0:
                        table_Q_simi[0].append(node)
                        table_Q_simi[1].append(Q_v)
                        # table_Q_simi[2].append(cid_simi[node])
                        # table_Q_simi[3].append(0)

                if len(table_Q_simi[0]) != 0:
                    if len(table_Q_simi[0]) == 1:
                        cid = table_Q_simi[0][0]
                    else:
                        # 通过函数计算P，然后轮盘赌得到cid
                        import numpy as np
                        table_Q_simi[1] = np.array(table_Q_simi[1]).reshape(-1, 1)
                        # table_Q_simi[2] = np.array(table_Q_simi[2]).reshape(-1, 1)
                        from sklearn import preprocessing
                        min_max_scaler = preprocessing.MaxAbsScaler()
                        # 用min-max归一化
                        tmp1 = min_max_scaler.fit_transform(table_Q_simi[1])
                        # tmp2 = min_max_scaler.fit_transform(table_Q_simi[2])
                        # table_Q_simi[1] = min_max_scaler.fit_transform(table_Q_simi[1])
                        # table_Q_simi[2] = min_max_scaler.fit_transform(table_Q_simi[2])

                        # tmp = (table_Q_simi[1]+self.lamb*table_Q_simi[2]).reshape(1,-1).tolist()[0]
                        tmp = tmp1.reshape(1, -1).tolist()[0]
                        eps = 10 ** (-8)
                        tmp = np.clip(tmp, eps, np.inf)

                        from tool import roulette
                        idx = roulette(tmp, seed)

                        cid = table_Q_simi[0][idx]
                        # print("点id：{};原社团：{}；新社团：{}".format(v_vid, v_cid, cid))
                    if cid != v_cid:
                        self._vid_vertex[v_vid]._cid = cid
                        # 在该社区编号下添加该节点
                        self._cid_vertices[cid].add(v_vid)
                        # 以前的社区中去除该节点
                        self._cid_vertices[v_cid].remove(v_vid)
                        # 模块度还能增加 继续迭代
                        # can_stop = False  # 由于不是每次都局部最大 会产生多跳循环导致死循环
                        mod_inc = True

                # if max_delta_Q > 0.0 and cid != v_cid:
                #     # 让该节点的社区编号变为取得最大增益邻居节点的编号
                #     self._vid_vertex[v_vid]._cid = cid
                #     # 在该社区编号下添加该节点
                #     self._cid_vertices[cid].add(v_vid)
                #     # 以前的社区中去除该节点
                #     self._cid_vertices[v_cid].remove(v_vid)
                #     # 模块度还能增加 继续迭代
                #     can_stop = False
                #     mod_inc = True
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
        self._cid_vertices = cid_vertices  # 相当于只记录了nodeId
        self._vid_vertex = vid_vertex  # 每个commId对应了原本的哪些nodes
        self._G = G  # 记录nodes(comms)之间的链接（邻居）

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
        label = [0] * self.numberOfnodes
        cid = -1
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                cid += 1
                for vid in vertices:
                    for idx in self._vid_vertex[vid]._nodes:
                        label[idx] = cid
        return label

    def execute(self, seed=0):
        # print("execute!!!!!!!!!!!!")
        iter_time = 1
        while True:
            iter_time += 1
            # 反复迭代，直到网络中任何节点的移动都不能再改善总的 modularity 值为止
            mod_inc = self.first_stage(seed)
            if mod_inc:
                self.second_stage()
            else:
                break
        return self.get_labels()


def evaluate(true_label, label):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    return nmi(true_label, label)


def main_qattack():
    name = 'polbooks'
    lamb = 1
    G = load_graph('../network/{}-qattack.txt'.format(name))
    algorithm = mod_Louvain(G, lamb)
    labels = algorithm.execute()
    import pickle
    label_path = "..//network//{}.label".format(name)
    with open(label_path, 'rb') as f:
        true_label = pickle.load(f)
    NMI = evaluate(true_label, labels)
    print(NMI)


def main_random():
    name = "cora"
    ratio = 0
    part = 0
    lamb = 1
    G = load_graph("..//network//{}//{}-ratio_{}-part{}.txt".format(name, name, ratio, part))
    algorithm = mod_Louvain(G, lamb)
    labels = algorithm.execute()
    import pickle
    label_path = "..//data//{}.label".format(name)
    with open(label_path, 'rb') as f:
        true_label = pickle.load(f)
    NMI = evaluate(true_label, labels)
    print(NMI)


def delErrNode_evaluate(true_label, labels, G):
    """被攻击后的图会出现孤立的点，在计算NMI时需要去掉"""
    """得到哪些点是孤立的"""
    del_indexs = set([i for i in range(len(true_label))])
    save_indexs = list(del_indexs & set(G.keys()))
    del_true_label = [true_label[idx] for idx in save_indexs]
    del_label = [labels[idx] for idx in save_indexs]
    NMI = evaluate(del_true_label, del_label)
    print(NMI)


def delErrNode(true_label, ec_labels, G):
    """
    根据给定的真实标签和Louvain算法生成的标签和图G，
    删除真实标签和算法生成标签中图中不存在的节点
    """
    del_indexs = set([i for i in range(len(true_label))])
    save_indexs = sorted(list(del_indexs & set(G.keys())))
    del_true_label = [true_label[idx] for idx in save_indexs]
    del_ec_labels = []
    for labels in ec_labels:
        del_ec_labels.append([labels[idx] for idx in save_indexs])
    return del_true_label, del_ec_labels


if __name__ == '__main__':
    main_random()

    # 更新总的边权重
    # for vid in self._G.keys():
    #     self._m += sum([self._G[vid][neighbor] for neighbor in self._G[vid].keys() if neighbor>vid])
