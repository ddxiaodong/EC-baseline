from louvain import mod_Louvain, load_graph, evaluate, load_graph_from_numpy

import pickle
from sklearn.metrics import normalized_mutual_info_score as nmi
import numpy as np
import random
from clusterEnsemble import hbgf, nmf, nmf_sklearn, org_nmf
from EC_tool import cal_simi

"""
实验文件
"""
def get_res(G, lamb, true_label, k, classNum, dataset=None, j_k=2, seed=0, noise_ratio=0):
    """
        获取结果并计算NMI、模块度和ARI
        参数：
        G ： 图字典
        lamb：模型参数
        true_label：真实的标签列表
        k：第一步骤生成k个结果
        classNum:类别数
        dataset:图名称
        j_k: Jaccard 相似度阶数
        seed: 随机种子

        返回：NMI得分、模块度、ARI得分
    """
    # algorithm = mod_Louvain(G, lamb)
    np.random.seed(seed) 
    seeds = np.random.randint(0, high=100, size=k, dtype='l')

    ec_labels = []  # 存储聚类结果
    clearN = len(true_label)  # 真实标签的数量

    if dataset != None:
        """加载相似度矩阵避免重复实验中的计算"""
        import os
        simi_save_root = "..//network_simi//"
        simi_save_name = "{}.simi-{}".format(dataset, j_k)
        simi_path = os.path.join(simi_save_root, simi_save_name)
        #  如果文件中存在就加载相似度矩阵，否则就计算相似度并保存
        if os.path.exists(simi_path):
            with open(simi_path, 'rb') as f:
                simi = pickle.load(f)
        else:
            simi = cal_simi(G, k=j_k)
            with open(simi_path, 'wb') as f:
                pickle.dump(simi, f)    # 保存数据
    else:
        simi = cal_simi(G, k=j_k)
    # 进行聚类并存储结果
    from tqdm import tqdm    # 创建一个迭代器，显示循环的进度条
    # 生成k个结果
    for idx in tqdm(range(k)):
        if clearN == len(G):
            # 创建一个mod_Louvain对象并传参
            algorithm = mod_Louvain(G, lamb=lamb, simi=simi)
        else:
            algorithm = mod_Louvain(G, lamb=lamb, clearN=clearN, simi=simi)
        # algorithm = mod_Louvain(G, lamb)
        res = algorithm.execute(seeds[idx])  # 得到改进的Lou算法执行后的节点标签
        ec_labels.append(res)   # 添加到结果集中

    # ec_labels = [algorithm.execute(seeds[i]) for i in range(k)]  # 得到k个结果
    # 如果标签数量不等于图中节点数量，则进行节点错误处理
    if clearN != len(G):
        from louvain import delErrNode
        true_label, ec_labels = delErrNode(true_label, ec_labels, G)
        # print("长度为:{} {}".format(len(true_label), len(ec_labels[0])))
    
    # 利用聚类结果计算NMI、模块度和ARI
    celabel = nmf_sklearn(np.array(ec_labels), classNum, 0)  # 生成NMF聚类结果
    
    # 计算NMI
    NMI = evaluate(true_label, celabel)
    
    # 计算模块度
    import networkx as nx
    G_nx = nx.Graph(G)
    communities = {}
    for i, label in enumerate(celabel):
        if label not in communities:
            communities[label] = []
        communities[label].append(i)
    modularity = nx.community.modularity(G_nx, communities.values())
    
    # 计算ARI
    from sklearn.metrics import adjusted_rand_score
    ARI = adjusted_rand_score(true_label, celabel)
    
    print("{} 噪声比例为{} 的实验结果：NMI={}, Modularity={}, ARI={}".format(dataset, noise_ratio, NMI, modularity, ARI))
    return NMI, modularity, ARI


def main_qattack(name='dblp', lamb=1, k=24, classNum=4, j_k=1, seed=0, noise_ratio=0):
    """
    
    参数:
    - name: 数据集名称
    - lamb: lambda参数
    - k: 生成k个结果
    - classNum: 类别数
    - j_k: Jaccard相似度阶数
    - seed: 随机种子
    - noise_ratio: 噪声比例
    返回: NMI得分
    """
    labelPath = '..//network//{}//{}.label'.format(name, name)
    with open(labelPath, 'rb') as f:
        true_label = pickle.load(f)
    
    # 修改为读取.npy邻接矩阵
    adj_matrix_path = f'../network/{name}/adj_matrix_{noise_ratio}.npy'
    G = load_graph_from_numpy(adj_matrix_path)
    
    graphName = "{}-qattack".format(name)
    return get_res(G=G, lamb=lamb, true_label=true_label, k=k, classNum=classNum, graphName=graphName, j_k=j_k, s=seed)


def main_attack(dataset='karate', noise_ratio=0, lamb=1, k=24, classNum=3, j_k=0, seed=0):
    """
    返回算法执行后的NMI得分
    参数:
    - dataset: 数据集名称 
    - noise_ratio: 噪声比例
    - lamb: Louvain算法超参数
    - k: 算法第一步执行k次生成k个聚类结果
    - classNum: 社区数量
    - j_k: 相似度阶数
    - seed: 随机种子
    - noise_ratio: 噪声比例
    """
    import pickle

    true_label = np.load(f"../{dataset}/labels.npy")
    # 修改为读取.npy邻接矩阵
    if (noise_ratio != 0):
        adj_matrix_path = f'../{dataset}/adj_matrix_{noise_ratio}.npy'
    else:
        adj_matrix_path = f'../{dataset}/adj_matrix.npy'

    G = load_graph_from_numpy(adj_matrix_path)
    
    return get_res(G=G, lamb=lamb, true_label=true_label, k=k, classNum=classNum, dataset=dataset, j_k=j_k, seed=seed, noise_ratio=noise_ratio)


def main():
    noise_ratio = 0.1
    # 对比实验设置
    setting = {
        "football": {'params': [noise_ratio, 0.1, 48, 12, 2, 0]},  # [noise_ratio, lamb, k, classNum, j_k, seed]
        "polbooks": {'params': [noise_ratio, 0.1, 24, 3, 2, 1]},
        "cora": {'params': [noise_ratio, 1, 12, 7, 1, 0]},
        "citeseer": {'params': [noise_ratio, 1, 24, 6, 2, 0]},
        "karate": {'params': [noise_ratio, 1, 24, 2, 2, 0]},
        "pubmed": {'params': [noise_ratio, 0, 24, 0, 0, 0]}
    }

    res = {}

    # 遍历数据集
    for data_name, data_setting in setting.items():
        params = data_setting['params']
        tmp_res = main_attack(dataset=data_name, noise_ratio=params[0], lamb=params[1],
                            k=params[2], classNum=params[3], j_k=params[4], seed=params[5])
        res[data_name] = tmp_res

    # 打印结果
    for data_name, res in res.items():
        print(data_name, res)


if __name__ == "__main__":
    main()

