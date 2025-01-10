import numpy as np
from scipy import sparse
from scipy.linalg import eigh

def network_mashup(W_in, ndim=500, restart_prob=0.5, threshold=None):
    """
    单网络版本的 Mashup 算法实现
    
    参数:
        W_in: np.ndarray, 输入的邻接矩阵 (N x N)
        ndim: int, 降维后的维度，默认500
        restart_prob: float, 随机游走重启概率，默认0.5
        threshold: float, 输出矩阵的阈值，默认为None（不使用阈值）
        
    返回:
        W_out: np.ndarray, Mashup处理后的矩阵
    """
    # 确保输入是numpy数组
    W = np.array(W_in)
    n = len(W)
    
    # 移除自环
    W = W * (1 - np.eye(n))
    
    # 计算转移概率矩阵
    degrees = np.sum(np.abs(W), axis=1)
    degrees[degrees == 0] = 1  # 避免除零错误
    D_inv = sparse.diags(1.0 / degrees)
    P = D_inv @ W
    
    # 由于P是numpy数组，不需要toarray()
    # 随机游走与重启
    identity = np.eye(n)
    try:
        rwr_matrix = np.linalg.inv(
            identity - (1 - restart_prob) * P
        ) * restart_prob
    except np.linalg.LinAlgError:
        # 如果矩阵不可逆，使用伪逆
        rwr_matrix = np.linalg.pinv(
            identity - (1 - restart_prob) * P
        ) * restart_prob
    
    # 特征分解
    eigenvalues, eigenvectors = eigh(rwr_matrix)
    
    # 选择前ndim个最大特征值
    idx = np.argsort(eigenvalues)[::-1][:ndim]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 构建低维表示
    reduced_features = eigenvectors @ np.diag(np.sqrt(np.abs(eigenvalues)))
    
    # 计算相似度矩阵
    W_out = reduced_features @ reduced_features.T
    
    # 后处理
    W_out = W_out * (1 - np.eye(n))  # 移除自环
    W_out = (W_out + W_out.T) / 2    # 确保对称性
    
    # 如果指定了阈值，进行阈值处理
    if threshold is not None:
        W_out = (W_out > threshold).astype(float)
    
    # 归一化
    if not np.all(W_out == 0):
        W_out = (W_out - np.min(W_out)) / (np.max(W_out) - np.min(W_out))
    
    return W_out 