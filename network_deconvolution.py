import numpy as np
from scipy import sparse
from scipy.linalg import eigh

def network_deconvolution(W_in, beta=0.99, alpha=1.0, control=0):
    """
    网络解卷积算法的Python实现（对称网络版本）
    
    参数:
        W_in: np.ndarray, 输入的相关性矩阵 (N x N)
        beta: float, 缩放参数，范围(0,1)
        alpha: float, 保留边的比例，范围(0,1]
        control: int, 是否显示非观察到的交互作用的直接权重
        
    返回:
        W_out: np.ndarray, 解卷积后的矩阵
    """
    # 移除自环
    n = len(W_in)
    W = W_in * (1 - np.eye(n))
    
    # 阈值处理
    y = np.quantile(W.flatten(), 1-alpha)
    W_th = W * (W >= y)
    
    # 确保对称性
    W_th = (W_th + W_th.T) / 2
    
    # 特征分解
    eigenvalues, eigenvectors = eigh(W_th)
    
    # 计算缩放因子
    lam_n = abs(min(min(eigenvalues), 0))
    lam_p = abs(max(max(eigenvalues), 0))
    m1 = lam_p * (1-beta) / beta
    m2 = lam_n * (1+beta) / beta
    m = max(m1, m2)
    
    # 网络解卷积
    d = eigenvalues / (m + eigenvalues)
    W_new = eigenvectors @ np.diag(d) @ eigenvectors.T
    
    # 处理直接权重
    if control == 0:
        ind_edges = (W_th > 0).astype(float)
        ind_nonedges = (W_th == 0).astype(float)
        m1 = np.max(W * ind_nonedges)
        m2 = np.min(W_new)
        W_new = (W_new + max(m1-m2, 0)) * ind_edges + (W * ind_nonedges)
    else:
        m2 = np.min(W_new)
        W_new = W_new + max(-m2, 0)
    
    # 归一化
    W_out = (W_new - np.min(W_new)) / (np.max(W_new) - np.min(W_new))
    
    return W_out

def network_deconvolution_regulatory(W_in, beta=0.5, alpha=0.1, control_p=0):
    """
    网络解卷积算法的Python实现（基因调控网络版本）
    
    参数:
        W_in: np.ndarray, 输入矩阵 (n_tf x n)，其中前n_tf个基因是转录因子
        beta: float, 缩放参数，范围(0,1)
        alpha: float, 保留边的比例，范围(0,1]
        control_p: int, 是否添加扰动以处理非对角化矩阵
        
    返回:
        W_out: np.ndarray, 解卷积后的矩阵
    """
    n_tf, n = W_in.shape
    
    # 移除自环
    W = W_in.copy()
    for i in range(n_tf):
        W[i,i] = 0
        
    # 使TF-TF网络对称
    tf_net = W[:n_tf, :n_tf]
    # 找到不对称的位置
    xx, yy = np.where(tf_net != tf_net.T)
    tf_net_final = tf_net.copy()
    
    # 处理不对称的情况
    for i in range(len(xx)):
        if tf_net[xx[i],yy[i]] != 0 and tf_net[yy[i],xx[i]] != 0:
            # 两个方向都有值，取平均
            avg_val = (tf_net[xx[i],yy[i]] + tf_net[yy[i],xx[i]]) / 2
            tf_net_final[xx[i],yy[i]] = avg_val
            tf_net_final[yy[i],xx[i]] = avg_val
        elif tf_net[xx[i],yy[i]] == 0:
            # 一个方向为0，使用另一个方向的值
            tf_net_final[xx[i],yy[i]] = tf_net[yy[i],xx[i]]
            tf_net_final[yy[i],xx[i]] = tf_net[yy[i],xx[i]]
        elif tf_net[yy[i],xx[i]] == 0:
            # 另一个方向为0，使用这个方向的值
            tf_net_final[xx[i],yy[i]] = tf_net[xx[i],yy[i]]
            tf_net_final[yy[i],xx[i]] = tf_net[xx[i],yy[i]]
    
    W[:n_tf, :n_tf] = tf_net_final
    
    # 设置网络密度为alpha
    y = np.quantile(W.flatten(), 1-alpha)
    W_th = W * (W >= y)
    
    # 确保TF-TF网络部分对称
    W_th[:n_tf, :n_tf] = (W_th[:n_tf, :n_tf] + W_th[:n_tf, :n_tf].T) / 2
    temp_net = (W_th > 0).astype(float)
    temp_net_remain = (W_th == 0).astype(float)
    W_th_remain = W * temp_net_remain
    m11 = np.max(W_th_remain)
    
    # 填充零使其成为方阵
    mat1 = np.zeros((n, n))
    mat1[:n_tf, :] = W_th
    
    # 检查矩阵是否可对角化
    if control_p != 1:
        try:
            U, D = np.linalg.eig(mat1)
            if np.linalg.cond(U) > 1e10:
                control_p = 1
        except np.linalg.LinAlgError:
            control_p = 1
    
    # 如果矩阵不可对角化，添加随机扰动
    if control_p == 1:
        r_p = 0.001
        rand_tf = r_p * np.random.rand(n_tf, n_tf)
        rand_tf = (rand_tf + rand_tf.T) / 2
        np.fill_diagonal(rand_tf, 0)
        
        rand_target = r_p * np.random.rand(n_tf, n-n_tf)
        mat_rand = np.hstack([rand_tf, rand_target])
        W_th = W_th + mat_rand
        
        mat1 = np.zeros((n, n))
        mat1[:n_tf, :] = W_th
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(mat1)
    
    # 基于特征值的缩放
    lam_n = abs(min(min(eigenvalues), 0))
    lam_p = abs(max(max(eigenvalues), 0))
    m1 = lam_p * (1-beta) / beta
    m2 = lam_n * (1+beta) / beta
    scale_eigen = max(m1, m2)
    
    # 应用网络解卷积滤波器
    D = np.diag(eigenvalues / (scale_eigen + eigenvalues))
    
    # 重构网络
    net_new = eigenvectors @ D @ np.linalg.inv(eigenvectors)
    net_new2 = net_new[:n_tf, :]
    m2 = np.min(net_new2)
    net_new3 = (net_new2 + max(m11-m2, 0)) * temp_net
    W_out = net_new3 + W_th_remain
    
    return W_out