import os
import warnings
from typing import Optional
import numpy as np
import pymetis  # 用阿里源conda装
from scipy import sparse
from sklearn.metrics import pairwise_distances, normalized_mutual_info_score
from sklearn.utils.extmath import safe_sparse_dot
import community
import networkx as nx

"""
实现共识图聚类的方法
"""
def create_hypergraph(base_clusters):
    """Create the incidence matrix of base clusters' hypergraph

    Parameter
    ----------
    base_clusters: labels generated by base clustering algorithms

    Return
    -------
    H: incidence matrix of base clusters' hypergraph
    """
    H = []
    len_bcs = base_clusters.shape[1]

    for bc in base_clusters:
        bc = np.nan_to_num(bc, nan=float('inf')) # [类别...节点数]
        unique_bc = np.unique(bc)
        len_unique_bc = len(unique_bc)  # 单个结果中聚类的类别数量
        bc2id = dict(zip(unique_bc, np.arange(len_unique_bc))) # {标签号:0到类别数-1}
        tmp = [bc2id[bc_elem] for bc_elem in bc] # 相当于统一到0到类别数-1
        h = np.identity(len_unique_bc, dtype=int)[tmp] # identity创建方矩阵（类别数*类别数）
        if float('inf') in bc2id.keys():
            h = np.delete(h, obj=bc2id[float('inf')], axis=1)
        H.append(sparse.csc_matrix(h))

    return sparse.hstack(H)
def to_pymetis_format(adj_mat):
    """Transform an adjacency matrix into the pymetis format
    将邻接矩阵转换为适用于 pymetis库的形式
    Parameter
    ---------
    adj_mat: adjacency matrix

    Returns
    -------
    xadj, adjncy, eweights: parameters for pymetis
    """
    xadj = [0]
    adjncy = []
    eweights = []
    n_rows = adj_mat.shape[0]
    adj_mat = adj_mat.tolil()

    for i in range(n_rows):
        row = adj_mat.getrow(i)
        idx_row, idx_col = row.nonzero()
        val = row[idx_row, idx_col]
        adjncy += list(idx_col)
        eweights += list(val.toarray()[0])
        xadj.append(len(adjncy))

    return xadj, adjncy, eweights
'''看懂修改处，这个是二部图+图切割'''
def hbgf(base_clusters, nclass):
    """Hybrid Bipartite Graph Formulation (HBGF)
    实现了共识聚类算法
    Parameters
    ----------
    base_clusters: 由基础聚类算法生成的标签集合
    nclass: 要生成的聚类数量

    Return
    -------
    celabel: 获得的共识聚类标签 from HBGF
    """
    A = create_hypergraph(base_clusters)
    rowA, colA = A.shape
    W = sparse.bmat([[sparse.dok_matrix((colA, colA)), A.T],
                     [A, sparse.dok_matrix((rowA, rowA))]])
    xadj, adjncy, _ = to_pymetis_format(W)
    membership = pymetis.part_graph(
        nparts=nclass, xadj=xadj, adjncy=adjncy, eweights=None)[1]
    celabel = np.array(membership[colA:])
    return celabel

'''nmf处理后续'''

def ecg_create_connectivity_matrix(base_clusters, edges, min_weight=0.05):
    #print(edges)

    # 这里edge要变成和del_label中一致的id
    ens_size =len(base_clusters)
    N = len(base_clusters[0])
    W = [0]*len(edges) # 边的权重
    ## Ensemble of level-1 Louvain
    for l in base_clusters:

        b = [l[x[0]]==l[x[1]] for x in edges]  # 遍历边，才统计是否在一个社团：只针对已有的边进行更新权重操作
        W = [W[i]+b[i] for i in range(len(W))]
    W = [min_weight + (1-min_weight)*W[i]/ens_size for i in range(len(W))]
    '''把边和w转为矩阵'''
    from scipy.sparse import csr_matrix
    if [edges[0][1], edges[0][0]] in edges:
        A = csr_matrix((W, ([edge[0] for edge in edges], [edge[1] for edge in edges])), shape=(N, N))
    else:
        #print([edge[0] for edge in edges])
        row =[edge[0] for edge in edges]
        row.extend([edge[1] for edge in edges])
        cor = [edge[1] for edge in edges]
        cor.extend([edge[0] for edge in edges])
        W.extend(W)
        #print(row)
        A = csr_matrix((W, (row, cor)), shape=(N, N))

    return A
def adv_ecg_create_connectivity_matrix(base_clusters, edges):
    min_weight = 0.05
    from EC_tool import cal_NMI
    ec_NMI_simi = cal_NMI(base_clusters)  # np.array()
    #print("NMI simi over")
    if (ec_NMI_simi==1).all():
        print("第一阶段随机性失效，直接输出结果")
    from org_louvain import Louvain
    # 由ec_NMI_simi创建G {u:{v1:1.0,v2}}
    # dis(xi，xj)≤max l{dis(xi，xl)，dis(xl, xj)}，l≠i,j
    G = create_RNG(ec_NMI_simi)
    #print("RNG over")
    algorithm = Louvain(G, )
    cluster_comm_res = algorithm.execute(False)
    #comm = np.array(cluster_comm_res, dtype=int)  #[[label1,label2...],[],...[]] 每个子列表代表一个社团
    _, len_bcs = base_clusters.shape
    #M = np.zeros((len_bcs, len_bcs))

    from scipy.sparse import csr_matrix
    if [edges[0][1], edges[0][0]] in edges:
        M = csr_matrix(([0.05]*len(edges), ([edge[0] for edge in edges], [edge[1] for edge in edges])), shape=(len_bcs, len_bcs)).A
    else:
        #print([edge[0] for edge in edges])
        row =[edge[0] for edge in edges]
        row.extend([edge[1] for edge in edges])
        cor = [edge[1] for edge in edges]
        cor.extend([edge[0] for edge in edges])
        W = [0.05]*len(edges)
        W.extend(W)
        #print(row)
        M = csr_matrix((W, (row, cor)), shape=(len_bcs, len_bcs)).A
    # min_weight + (1-min_weight)*W[i]/ens_size
    res_M = np.zeros_like(M)
    for in_base_clusters in cluster_comm_res:
        M += (1-min_weight)*(ecg_create_connectivity_matrix([base_clusters[idx] for idx in in_base_clusters], edges, 0).A)/len(cluster_comm_res)
    return sparse.csr_matrix(res_M+M)
def create_connectivity_matrix(base_clusters):
    """Create the connectivity matrix

    Parameter
    ---------
    base_clusters: labels generated by base clustering algorithms

    Return
    ------
    M: connectivity matrix
    """
    n_bcs, len_bcs = base_clusters.shape
    M = np.zeros((len_bcs, len_bcs))
    m = np.zeros_like(M)

    for bc in base_clusters:
        for i, elem_bc in enumerate(bc):
            m[i] = np.where(elem_bc == bc, 1, 0)
        M += m

    M /= n_bcs
    return sparse.csr_matrix(M)
def orthogonal_nmf_algorithm(W, nclass, random_state, maxiter):
    """Algorithm for bi-orthogonal three-factor NMF problem

    Parameters
    ----------
    W: given matrix
    random_state: used for reproducible results
    maxiter: maximum number of iterations

    Return
    -------
    Q, S: factor matrices
    """
    np.random.seed(random_state)

    n = W.shape[0]
    Q = np.random.rand(n, nclass).reshape(n, nclass)
    S = np.diag(np.random.rand(nclass))

    for _ in range(maxiter):
        # Update Q
        WQS = safe_sparse_dot(W, np.dot(Q, S), dense_output=True)
        Q = Q * np.sqrt(WQS / (np.dot(Q, np.dot(Q.T, WQS)) + 1e-8))
        # Update S
        QTQ = np.dot(Q.T, Q)
        WQ = safe_sparse_dot(W, Q, dense_output=False)
        QTWQ = safe_sparse_dot(Q.T, WQ, dense_output=True)
        S = S * np.sqrt(QTWQ / (np.dot(QTQ, np.dot(S, QTQ)) + 1e-8))

    return Q, S
def nmf(base_clusters, nclass, random_state, maxiter=200):
    """NMF-based consensus clustering

    Parameters
    ----------
    base_clusters: labels generated by base clustering algorithms
    nclass: number of classes
    random_state: used for reproducible results
    maxiter: maximum number of iterations

    Return
    -------
    celabel: consensus clustering label obtained from NMF
    """
    M = adv_create_connectivity_matrix(base_clusters)
    Q, S = orthogonal_nmf_algorithm(M, nclass, random_state, maxiter)
    celabel = np.argmax(np.dot(Q, np.sqrt(S)), axis=1)
    return celabel
def nmf_sklearn(base_clusters, nclass, random_state, maxiter=200, org_switch = False, org_ecg_switch=False, adv_ecg_switch=False, edges = None):
    """
    实现了基于sklearn库中NMF（Non-Negative Matrix Factorization）算法的聚类方法
    参数：
    base_clusters 基础聚类算法生成的标签集合
    nclass  聚类的类别数量
    random_state 控制随机数生成
    maxiter 最大迭代次数
    org_switch  布尔值，指示是否使用原始ECG连接矩阵
    org_ecg_switch 布尔值，指示是否使用改进后的ECG连接矩阵
    edges 边信息
    输出：
    聚类结果的标签
    """
    if org_switch:
        if org_ecg_switch:
            print("org ecg create")
            M = ecg_create_connectivity_matrix(base_clusters, edges)
        else:
            print("org create")
            M = create_connectivity_matrix(base_clusters)

    else:
        if adv_ecg_switch:
            print("adv ecg create")
            M = ecg_create_connectivity_matrix(base_clusters, edges)

        else:
            print("adv create")
            M = adv_create_connectivity_matrix(base_clusters)
    from sklearn.decomposition import NMF
    #print("start NMF")
    nmf_model = NMF(n_components=nclass,
                    init="random",
                    random_state=random_state,
                    #l1_ratio=1,
                    #alpha=0.5,
                    max_iter=maxiter, regularization='both')  # random_state=self.args.seed,

    U = nmf_model.fit_transform(sparse.csr_matrix(M),)
    #print('start getting membership.')
    celabel = np.argmax(U, axis=1)

    #celabel = np.argmax(np.dot(Q, np.sqrt(S)), axis=1)
    return celabel
def lou(base_clusters):
    M = adv_create_connectivity_matrix(base_clusters)
    from org_louvain import Louvain
    import collections
    A = M.A
    A_adj = collections.defaultdict(dict)
    n = A.shape[0]
    for r in range(n):
        for c in range(r + 1, n):
            if A[r][c] != 0:
                if r not in A_adj:
                    A_adj[r] = {c: A[r][c]}
                else:
                    A_adj[r][c] = A[r][c]
                if c not in A_adj:
                    A_adj[c] = {r: A[r][c]}
                else:
                    A_adj[c][r] = A[r][c]
    path = "cora-ec1.G"
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(A_adj, f)
    print("over")
    algorithm = Louvain(A_adj, )
    celabel = algorithm.execute()
    return celabel
def pyLOU(base_clusters, org_switch = False, org_ecg_switch=False, adv_ecg_switch=False, edges = None):
    if org_switch:
        if org_ecg_switch:
            print("org ecg create")
            M = ecg_create_connectivity_matrix(base_clusters, edges)
        else:
            print("org create")
            M = LOU_create_connectivity_matrix(base_clusters, edges)
            #M = create_connectivity_matrix(base_clusters, )
    else:
        if adv_ecg_switch:
            print("adv ecg create")

            M = adv_ecg_create_connectivity_matrix(base_clusters, edges)

        else:
            print("adv create")
            #M = adv_create_connectivity_matrix(base_clusters, )
            M = LOU_adv_create_connectivity_matrix(base_clusters, edges)
    graph = nx.from_scipy_sparse_matrix(M)
    print("grraph over")
    partition = community.best_partition(graph)
    print("comm over")
    org_LOU_label = [partition[idx] for idx in sorted(partition.keys())]
    return org_LOU_label
def org_nmf(base_clusters, nclass, random_state, maxiter=200):
    """NMF-based consensus clustering

    Parameters
    ----------
    base_clusters: labels generated by base clustering algorithms
    nclass: number of classes
    random_state: used for reproducible results
    maxiter: maximum number of iterations

    Return
    -------
    celabel: consensus clustering label obtained from NMF
    """
    M = create_connectivity_matrix(base_clusters)
    Q, S = orthogonal_nmf_algorithm(M, nclass, random_state, maxiter)
    celabel = np.argmax(np.dot(Q, np.sqrt(S)), axis=1)
    return celabel
def base_nmf(clear_G, nclass, random_state, maxiter=200):
    numberOfnodes = len(clear_G)
    A = np.zeros((numberOfnodes, numberOfnodes))
    for u, vs in clear_G.items():
        for v in vs.keys():
            A[u][v] = clear_G[u][v]
            #A[v][u] = clear_G[u][v]
    M = sparse.csr_matrix(A)
    Q, S = orthogonal_nmf_algorithm(M, nclass, random_state, maxiter)
    celabel = np.argmax(np.dot(Q, np.sqrt(S)), axis=1)
    return celabel
'''改进后的consensus graph创建'''
def create_RNG(simi_m):
    '''创建RNG图
    输入是一个相似度矩阵
    输出相关性网络图
    '''
    max_idx = simi_m.argmax(axis=1)  # 找到每行中相似度最大的数据点的索引 
    import collections
    G = collections.defaultdict(dict)
    for u, v in enumerate(max_idx): # 创建字典，对每个点将其与最佳邻居添加一条边 权重设为1 
        G[u][v] = 1.0
        G[v][u] = 1.0
    return G
def adv_create_connectivity_matrix(base_clusters):
    from EC_tool import cal_NMI
    ec_NMI_simi = cal_NMI(base_clusters)  # np.array()  计算相似度矩阵 
    #print("NMI simi over")
    if (ec_NMI_simi==1).all():
        print("第一阶段随机性失效，直接输出结果")
    from org_louvain import Louvain
    # 由ec_NMI_simi创建G {u:{v1:1.0,v2}}
    # dis(xi，xj)≤max l{dis(xi，xl)，dis(xl, xj)}，l≠i,j
    G = create_RNG(ec_NMI_simi) # 根据ec_NMI_simi创建一个相关性网络图G 每个节点代表一个基础聚类结果，节点之间的边权表示它们之间的NMI相似度
    #print("RNG over")
    algorithm = Louvain(G, )  #使用LOU算法对G执行
    cluster_comm_res = algorithm.execute(False) # 社团划分结果
    #comm = np.array(cluster_comm_res, dtype=int)  #[[label1,label2...],[],...[]] 每个子列表代表一个社团
    # 对每个社区内进行合并
    #in_res = []

    len_bcs = len(base_clusters[0])
    res_M = np.zeros((len_bcs, len_bcs))
    for idx, comm_in_idx in enumerate(cluster_comm_res):
        #社团内
        in_base_clusters = np.array([base_clusters[i] for i in comm_in_idx], dtype=int)
        n_bcs, len_bcs = in_base_clusters.shape
        M = np.zeros_like(res_M)
        m = np.zeros_like(M)
        #print("m")
        for bc in in_base_clusters:
            for i, elem_bc in enumerate(bc):
                m[i] = np.where(elem_bc == bc, 1, 0)
            M += m
        # mask = M==1  # 去除掉没有共识的边
        # M[mask]=0

        M /= n_bcs
        res_M += M
        #in_res.append(M)

    res_M[res_M<0.05] = 0
    # 对社团间进行合并
    #print("start sum")
    # res = np.zeros_like(in_res[0])
    # for adj in in_res:
    #     res+=adj
    #print("sum over")
    return sparse.csr_matrix(res_M)#sparse.csr_matrix(res) 得到的矩阵转换为稀疏矩阵并返回
    # return res_M
def LOU_adv_create_connectivity_matrix(base_clusters, edges):

    from tool import cal_NMI
    ec_NMI_simi = cal_NMI(base_clusters)  # np.array()
    if (ec_NMI_simi==1).all():
        print("第一阶段随机性失效，直接输出结果")
    from org_louvain import Louvain
    G = create_RNG(ec_NMI_simi)
    algorithm = Louvain(G, )
    cluster_comm_res = algorithm.execute(False)
    len_bcs = len(base_clusters[0])

    min_weight = 1/(len(cluster_comm_res))
    from scipy.sparse import csr_matrix
    if [edges[0][1], edges[0][0]] in edges:
        res_M = csr_matrix(([0.05]*len(edges), ([edge[0] for edge in edges], [edge[1] for edge in edges])), shape=(len_bcs, len_bcs)).A
    else:
        #print([edge[0] for edge in edges])
        row =[edge[0] for edge in edges]
        row.extend([edge[1] for edge in edges])
        cor = [edge[1] for edge in edges]
        cor.extend([edge[0] for edge in edges])
        W = [min_weight]*len(edges)
        W.extend(W)
        #print(row)
        res_M = csr_matrix((W, (row, cor)), shape=(len_bcs, len_bcs)).A


    #res_M = np.zeros((len_bcs, len_bcs))
    for idx, comm_in_idx in enumerate(cluster_comm_res):
        #社团内
        in_base_clusters = np.array([base_clusters[i] for i in comm_in_idx], dtype=int)
        n_bcs, len_bcs = in_base_clusters.shape
        M = np.zeros_like(res_M)
        m = np.zeros_like(M)
        #print("m")
        for bc in in_base_clusters:
            for i, elem_bc in enumerate(bc):
                m[i] = np.where(elem_bc == bc, 1, 0)
            M += m
        M /= n_bcs
        M[M<=1] = 0
        res_M += (1 - min_weight) * (M) / len(in_base_clusters)

    #res_M[res_M<0.5] = 0

    return sparse.csr_matrix(res_M)#sparse.csr_matrix(res)
def LOU_create_connectivity_matrix(base_clusters, edges):
    """Create the connectivity matrix

    Parameter
    ---------
    base_clusters: labels generated by base clustering algorithms

    Return
    ------
    M: connectivity matrix
    """
    min_weight = 1/(len(base_clusters))
    len_bcs = len(base_clusters[0])
    from scipy.sparse import csr_matrix
    if [edges[0][1], edges[0][0]] in edges:
        res_M = csr_matrix(([0.05]*len(edges), ([edge[0] for edge in edges], [edge[1] for edge in edges])), shape=(len_bcs, len_bcs)).A
    else:
        #print([edge[0] for edge in edges])
        row =[edge[0] for edge in edges]
        row.extend([edge[1] for edge in edges])
        cor = [edge[1] for edge in edges]
        cor.extend([edge[0] for edge in edges])
        W = [min_weight]*len(edges)
        W.extend(W)
        #print(row)
        res_M = csr_matrix((W, (row, cor)), shape=(len_bcs, len_bcs)).A

    n_bcs, len_bcs = base_clusters.shape
    M = np.zeros((len_bcs, len_bcs))
    m = np.zeros_like(M)

    for bc in base_clusters:
        for i, elem_bc in enumerate(bc):
            m[i] = np.where(elem_bc == bc, 1, 0)
        M += m

    M /= n_bcs
    M[M <=1] = 0
    res_M += (1 - min_weight) * (M)
    return sparse.csr_matrix(M)
if __name__ == "__main__":
    base_clusters = np.array([[1,1,2,2],[1,2,1,2],[3,3,3,3]])
    celabel = hbgf(base_clusters, 2)
    print(celabel)
    celabel2 = nmf(base_clusters, 2, 0)
    print(celabel2)