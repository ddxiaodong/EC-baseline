import numpy as np
from scipy.linalg import eig
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.datasets import Actor

# dataset = Planetoid(root='./cora', name='Cora')
# dataset = Planetoid(root='./citeseer', name='Citeseer')
# dataset = Actor("./Actor")




# ND 核心函数  输入邻接矩阵
def ND(mat, beta=0.99, alpha=1, control=0):
    if not (0 < beta < 1):
        raise ValueError('Error: beta should be in (0,1)')
    if not (0 < alpha <= 1):
        raise ValueError('Error: alpha should be in (0,1]')

    n = mat.shape[0]
    mat = mat * (1 - np.eye(n))

    y = np.quantile(mat.flatten(), 1 - alpha)
    mat_th = mat * (mat >= y)

    mat_th = (mat_th + mat_th.T) / 2   # 确保是对称阵

    # 进行矩阵特征值分解  并对特征值进行调整
    U, D = np.linalg.eig(mat_th)

    lam_n = abs(min(np.min(D), 0))
    lam_p = abs(max(np.max(D), 0))

    m1 = lam_p * (1 - beta) / beta
    m2 = lam_n * (1 + beta) / beta
    m = max(m1, m2)

    D_inv = np.diag(1 / (m + np.diag(D)))


    mat_new1 = U @ D_inv @ U.T

    if control == 0:
        ind_edges = (mat_th > 0).astype(float)
        ind_nonedges = (mat_th == 0).astype(float)
        m1 = np.max(mat * ind_nonedges)
        m2 = np.min(mat_new1)
        mat_new2 = (mat_new1 + np.maximum(m1 - m2, 0)) * ind_edges + (mat * ind_nonedges)
    elif control == 1:
        m2 = np.min(mat_new1)
        mat_new2 = mat_new1 + np.maximum(-m2, 0)

    # 假设 mat_new2 是经过调整后的矩阵
    m1 = np.min(mat_new2)
    m2 = np.max(mat_new2)
    # 线性映射到区间 [0, 1]
    mat_nd = (mat_new2 - m1) / (m2 - m1)

    return mat_nd

import community as community_louvain  # python-louvain 库
import networkx as nx

def community_detection_ND(denoised_matrix):

    # 将去噪后的矩阵转换为 NetworkX 图
    denoised_graph = nx.from_numpy_array(denoised_matrix)
    # 使用 Louvain 算法进行社团检测
    partition = community_louvain.best_partition(denoised_graph)

    # 输出社团检测结果
    print("社团检测结果：", partition)
    return partition

# 评估
def evaluate(partition, denoised_matrix, true_partition):
    denoised_graph = nx.from_numpy_array(denoised_matrix)
    # 计算模块度
    modularity = community_louvain.modularity(partition, denoised_graph)
    print("模块度：", modularity)

    # 如果有真实社团划分，计算 NMI
    from sklearn.metrics import normalized_mutual_info_score

    pred_partition = list(partition.values())  # 预测社团划分

    nmi = normalized_mutual_info_score(true_partition, pred_partition)
    print("标准化互信息（NMI）：", nmi)
    return nmi

# 读取数据文件并转换为ND可以识别的邻接矩阵形式
def read_data(file_path, label_path):
    """
    读取边列表文件并返回 NetworkX 图（无权重）
    """
    graph = nx.Graph()  # 创建无向图
    with open(file_path, 'r') as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())  # 读取节点1和节点2
            graph.add_edge(node1, node2)  # 添加边（无权重）
    """
    将 NetworkX 图转换为邻接矩阵
    """
    n = graph.number_of_nodes()  # 节点数
    print("节点数：", n)
    adj_matrix = nx.to_numpy_array(graph, nodelist=range(n))  # 转换为邻接矩阵

    """
    读取标签文件并返回标签列表
    """
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))  # 假设每行是一个整数标签
    # 示例：读取 actor.label

    print(" 标签数量：", len(labels))


    return adj_matrix, labels

def pyg_to_adj(data):
    # edge_index = data.edge_index
    # edge_index = edge_index.cpu().numpy()
    # n = data.num_nodes
    # adj_matrix = np.zeros((n, n))
    # for i in range(edge_index.shape[1]):
    #     adj_matrix[edge_index[0, i], edge_index[1, i]] = 1
    #     adj_matrix[edge_index[1, i], edge_index[0, i]] = 1
    # return adj_matrix
    # 创建邻接矩阵
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i, j in edge_index.T:
        adj_matrix[i, j] = 1

    # Make the adjacency matrix symmetric (since Cora is an undirected graph)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    return adj_matrix






def main():
    # 统一读取pyg框架中的数据吧 这样节点数量、边数量等都有一个固定的大小
    dataset = "cora"
    noise_ratio = 0.1
    filename = "{}_random_{}".format(dataset, noise_ratio)


    data_pyg = torch.load(f'..//network//{dataset}/{filename}.pt')
    data_pyg = data_pyg.to('cpu')
    # 不管什么数据 都转换为函数需要的形式
    adj_matrix = pyg_to_adj(data_pyg)
    # 原始的数据集评估
    ori_partition = community_detection_ND(adj_matrix)
    labels = data_pyg.y.numpy()
    evaluate(ori_partition, adj_matrix, labels)
    # 使用ND去噪
    denoised_matrix = ND(adj_matrix)
    # 确保邻接矩阵是实数（ND处理后会出现复数部分，不能进行社团划分）
    if np.iscomplexobj(denoised_matrix):
        denoised_matrix = np.real(denoised_matrix)  # 取实部
    partition = community_detection_ND(denoised_matrix)
    labels = data_pyg.y.numpy()
    evaluate(partition, denoised_matrix, labels)




if __name__ == '__main__':
    main()