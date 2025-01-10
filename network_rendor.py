import numpy as np
from scipy import sparse
from scipy.linalg import eigh

def partial(mat):
    """部分标准化矩阵
    
    参数:
        mat: np.ndarray, 输入矩阵
        
    返回:
        np.ndarray: 标准化后的矩阵
    """
    diag = np.diag(mat)
    diag_inv = np.diag(1.0 / diag) 
    mat_new = -diag_inv @ mat @ diag_inv
    np.fill_diagonal(mat_new, 1)
    return mat_new

def network_rendor(W_in, m=2, eps1=1, eps2=1):
    """RENDOR (Reverse Network Diffusion to Remove indirect noise)算法
    
    参数:
        W_in: np.ndarray, 输入的邻接矩阵 (N x N)
        m: float, 扩散步长参数，默认为2
        eps1: float, 第一个正则化参数，默认为1
        eps2: float, 第二个正则化参数，默认为1
        
    返回:
        np.ndarray: 去噪后的矩阵
    """
    W = np.array(W_in)
    n = len(W)
    W = W + W.T
    
    W = W + eps1 + eps2 * np.eye(n)
    
    P1 = W / W.sum(axis=1)[:, np.newaxis]
    
    P2 = m * np.linalg.inv((m-1) * np.eye(n) + P1) @ P1
    P2 = P2 - np.minimum(P2.min(axis=1), 0)[:, np.newaxis]
    P2 = P2 / P2.sum(axis=1)[:, np.newaxis]
    
    W_out = np.diag(W.sum(axis=1)) @ P2
    
    W_out = (W_out - W_out.min()) / (W_out.max() - W_out.min())
    W_out = W_out + W_out.T
    np.fill_diagonal(W_out, 0)
    
    return W_out

def network_icm(W_in):
    """ICM (Inverse Correlation Matrix)算法
    
    参数:
        W_in: np.ndarray, 输入的邻接矩阵 (N x N)
        
    返回:
        np.ndarray: 去噪后的矩阵
    """
    W = np.array(W_in)
    n = W.shape[0]
    W = W + W.T
    
    W_new = np.linalg.pinv(W)
    W_new = W_new + 1e-18
    W_new = partial(W_new)
    W_new = np.abs(W_new)
    
    W_new = (W_new - W_new.min()) / (W_new.max() - W_new.min())
    W_new = W_new + W_new.T
    np.fill_diagonal(W_new, 0)
    
    return W_new

def network_silencer(W_in):
    """Silencer去噪算法
    
    参数:
        W_in: np.ndarray, 输入的邻接矩阵 (N x N)
        
    返回:
        np.ndarray: 去噪后的矩阵
    """
    W = np.array(W_in)
    n = W.shape[0]
    W = W + W.T
    
    temp = np.diag(np.diag((W - np.eye(n)) @ W))
    W_new = (W - np.eye(n) + temp) @ np.linalg.pinv(W)
    W_new = np.abs(W_new)
    
    W_new = (W_new - W_new.min()) / (W_new.max() - W_new.min())
    W_new = W_new + W_new.T
    np.fill_diagonal(W_new, 0)
    
    return W_new

def network_nd(W_in):
    """Network Deconvolution (ND)算法
    
    参数:
        W_in: np.ndarray, 输入的邻接矩阵 (N x N)
        
    返回:
        np.ndarray: 去噪后的矩阵
    """
    W = np.array(W_in)
    n = W.shape[0]
    W = W + W.T
    
    eigenvals, eigenvecs = np.linalg.eigh(W)
    D = np.diag(eigenvals)
    U = eigenvecs
    
    lam_n = abs(min(min(np.diag(D)), 0))
    lam_p = abs(max(max(np.diag(D)), 0))
    
    beta = 0.99
    m1 = lam_p * (1-beta)/beta
    m2 = lam_n * (1+beta)/beta
    m = max(m1, m2)
    
    D = np.diag(np.diag(D)/(m + np.diag(D)))
    W_nd = U @ D @ np.linalg.pinv(U)
    
    W_nd = (W_nd - W_nd.min()) / (W_nd.max() - W_nd.min())
    W_nd = W_nd + W_nd.T
    np.fill_diagonal(W_nd, 0)
    
    return W_nd 