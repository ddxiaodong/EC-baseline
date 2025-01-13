import numpy as np
from scipy import sparse
from scipy.linalg import eigh

def network_enhancement(W_in, order=2, K=None, alpha=0.9):
    """
    网络增强算法的Python实现
    
    参数:
        W_in: np.ndarray, 输入的邻接矩阵 (N x N)
        order: float, 扩散程度，典型值为0.5、1、2
        K: int, 邻居数量
        alpha: float, 正则化参数
        
    返回:
        W_out: np.ndarray, 增强后的邻接矩阵
    """
    
    # 设置默认参数
    if K is None:
        K = min(20, int(np.ceil(len(W_in)/10)))
    
    # 移除自环
    W_in1 = W_in * (1 - np.eye(len(W_in)))
    
    # 找到非零行/列的索引
    zeroindex = np.where(np.sum(np.abs(W_in1), axis=1) > 0)[0]
    W0 = W_in[np.ix_(zeroindex, zeroindex)]
    
    # 度归一化
    W = network_diffusion(W0, diffusion_type='ave')
    W = (W + W.T) / 2
    
    DD = np.sum(np.abs(W0), axis=1)
    
    # 如果是二值矩阵，直接使用
    if len(np.unique(W.flatten())) == 2:
        P = W
    else:
        # 获取主导集
        P = dominate_set(np.abs(W), min(K, len(W)-1)) * np.sign(W)
    
    # 添加自环
    P = P + (np.eye(len(P)) + np.diag(np.sum(np.abs(P.T), axis=0)))
    
    # 转换为转移矩阵
    P = transition_fields(P)
    
    # 特征分解
    eigenvalues, eigenvectors = eigh(P)
    
    # 谱变换
    d = eigenvalues - np.finfo(float).eps
    d = (1 - alpha) * d / (1 - alpha * d**order)
    
    # 重构矩阵
    W = eigenvectors @ np.diag(d) @ eigenvectors.T
    W = W * (1 - np.eye(len(W))) / (1 - np.diag(W))[:, None]
    
    # 度缩放
    D = sparse.diags(DD)
    W = D @ W
    
    # 后处理
    W[W < 0] = 0
    W = (W + W.T) / 2
    
    # 构建输出矩阵
    W_out = np.zeros_like(W_in)
    W_out[np.ix_(zeroindex, zeroindex)] = W
    
    return W_out

def network_diffusion(W, diffusion_type='ave'):
    """
    网络扩散
    
    参数:
        W: np.ndarray, 输入矩阵
        diffusion_type: str, 扩散类型,'ave'或'gph'
        
    返回:
        W_normalized: np.ndarray, 扩散后的矩阵
    """
    W = W * len(W)
    D = np.sum(np.abs(W), axis=1) + np.finfo(float).eps
    
    if diffusion_type == 'ave':
        D = 1.0 / D
        D = sparse.diags(D)
        W_normalized = D @ W
    elif diffusion_type == 'gph':
        D = 1.0 / np.sqrt(D)
        D = sparse.diags(D)
        W_normalized = D @ (W @ D)
    
    return W_normalized

def dominate_set(aff_matrix, k):
    """
    计算主导集
    
    参数:
        aff_matrix: np.ndarray, 亲和度矩阵
        k: int, 每个节点保留的最大边数
        
    返回:
        np.ndarray: 主导集矩阵
    """
    n = len(aff_matrix)
    # 对每行排序并取前k个最大值
    sorted_indices = np.argsort(-aff_matrix, axis=1)[:, :k]
    sorted_values = np.take_along_axis(aff_matrix, sorted_indices, axis=1)
    
    # 构建稀疏矩阵
    rows = np.repeat(np.arange(n)[:, None], k, axis=1).flatten()
    cols = sorted_indices.flatten()
    vals = sorted_values.flatten()
    
    # 构建对称矩阵
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    matrix = (matrix + matrix.T) / 2
    
    return matrix.toarray()

def transition_fields(W):
    """
    转换为转移矩阵
    
    参数:
        W: np.ndarray, 输入矩阵
        
    返回:
        np.ndarray: 转移矩阵
    """
    zeroindex = np.where(np.sum(W, axis=1) == 0)[0]
    W = W * len(W)
    W = network_diffusion(W, 'ave')
    
    # 归一化
    w = np.sqrt(np.sum(np.abs(W), axis=0) + np.finfo(float).eps)
    W = W / w
    W = W @ W.T
    
    # 处理零索引
    W[zeroindex, :] = 0
    W[:, zeroindex] = 0
    
    return W 