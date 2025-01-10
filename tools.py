import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from urllib.request import urlretrieve
import os






def analyze_results(original_matrix, deconv_matrix, labels):
    """
    分析原始邻接矩阵和解卷积后的邻接矩阵,并进行社团检测
    
    参数:
        original_matrix: 原始邻接矩阵
        deconv_matrix: 解卷积后的邻接矩阵
        
    返回:
        original_communities: 原始网络的社团划分结果
        deconv_communities: 解卷积网络的社团划分结果
    """
    import networkx as nx
    from community import community_louvain
    import matplotlib.pyplot as plt
    from sklearn.metrics.cluster import normalized_mutual_info_score
    
    # 将邻接矩阵转换为NetworkX图
    G_original = nx.from_numpy_array(original_matrix)
    G_deconv = nx.from_numpy_array(deconv_matrix)
    
    # 使用Louvain算法进行社团检测
    original_communities = community_louvain.best_partition(G_original, resolution=2.0)
    deconv_communities = community_louvain.best_partition(G_deconv, resolution=2.0)
    
    # 计算模块度
    original_modularity = community_louvain.modularity(original_communities, G_original)
    deconv_modularity = community_louvain.modularity(deconv_communities, G_deconv)
    # 计算NMI
    original_labels = [original_communities[i] for i in range(len(original_communities))]
    deconv_labels = [deconv_communities[i] for i in range(len(deconv_communities))]
    
    nmi_original = normalized_mutual_info_score(labels, original_labels)
    nmi_deconv = normalized_mutual_info_score(labels, deconv_labels)

    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    # 绘制原始网络社团结构
    plt.subplot(121)
    pos_original = nx.spring_layout(G_original)
    nx.draw_networkx_nodes(G_original, pos_original, node_color=list(original_communities.values()),
                          cmap=plt.cm.rainbow, node_size=20)
    nx.draw_networkx_edges(G_original, pos_original, alpha=0.2)
    plt.title(f'origin_network_structure\nmodularity: {original_modularity:.3f}')
    
    # 绘制解卷积后网络社团结构
    plt.subplot(122)
    pos_deconv = nx.spring_layout(G_deconv)
    nx.draw_networkx_nodes(G_deconv, pos_deconv, node_color=list(deconv_communities.values()),
                          cmap=plt.cm.rainbow, node_size=20)
    nx.draw_networkx_edges(G_deconv, pos_deconv, alpha=0.2)
    plt.title(f'deconvolution_network_structure\nmodularity: {deconv_modularity:.3f}')
    
    plt.tight_layout()
    plt.show()
    
    # 打印社团统计信息
    print(f'原始网络社团数量: {len(set(original_communities.values()))}')
    print(f'解卷积后网络社团数量: {len(set(deconv_communities.values()))}')
    print(f'原始网络的模块度：{original_modularity:.3f}')
    print(f'解卷积后网络的模块度：{deconv_modularity:.3f}')
    print(f'原始网络NMI: {nmi_original:.3f}')
    print(f'解卷积后网络NMI: {nmi_deconv:.3f}')
    
    return original_communities, deconv_communities



