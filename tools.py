import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import os
import networkx as nx
from community import community_louvain
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

def analyze_results(adj_matrix, labels):
    """
    分析网络的社团结构和评估指标
    
    参数:
        adj_matrix: np.ndarray, 邻接矩阵
        labels: list/array, 节点的真实标签
        
    返回:
        communities: dict, 社团划分结果
        modularity: float, 模块度
        nmi: float, 标准化互信息
        ari: float, 调整兰德系数
    """

    
    # 将邻接矩阵转换为NetworkX图
    G = nx.from_numpy_array(adj_matrix)
    
    # 使用Louvain算法进行社团检测
    communities = community_louvain.best_partition(G, resolution=2.0)
    
    # 计算模块度
    modularity = community_louvain.modularity(communities, G)
    
    # 获取社团标签列表
    pred_labels = [communities[i] for i in range(len(communities))]
    
    # 计算NMI和ARI
    nmi = normalized_mutual_info_score(labels, pred_labels)
    ari = adjusted_rand_score(labels, pred_labels)

    return communities, modularity, nmi, ari
