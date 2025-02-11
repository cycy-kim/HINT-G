import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse import csr_matrix



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        values = values.astype(np.float32)
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def get_criterion(criterion_name):
    """"""


def distance_feature(feat_1, feat_2):
    pdist = nn.PairwiseDistance(p=2)
    output = pdist(feat_1, feat_2) /4
    return output

def distance_graph_prob(adj_1, adj_2_prob):
    dist = F.binary_cross_entropy(adj_2_prob, adj_1)
    return dist

def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    # return torch.tensor(vector)
    return vector.clone().detach().requires_grad_(True)

def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows).to(vector.device)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def get_degree_matrix(adj):
    return torch.diag(torch.sum(adj.clone().detach(), dim=1))

def shifted_sigmoid(x, shift=3):
    return 1 / (1 + torch.exp(-(x - shift)))

# 이거 x(features), adj 이렇게 tuple로 반환하게 바꿔야됨
# def mask_fn(data: Data, node_mask: np.array):
# 	""" subgraph building through spliting the selected nodes from the original graph """
# 	row, col = data.edge_index
# 	edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
# 	ret_edge_index = data.edge_index[:, edge_mask]
# 	ret_edge_attr = None if data.edge_attr is None else data.edge_attr[edge_mask] 
# 	data = Data(x=data.x, edge_index=ret_edge_index, 
# 		edge_attr=ret_edge_attr, batch=data.batch)
# 	return data

def mask_fn(graph: tuple, mask: np.array):
    x, adj = graph
    node_indices = np.where(mask)[0]
    new_x = x[node_indices]
    
    new_adj = adj[node_indices][:, node_indices]
    
    return new_x, new_adj





"""
stat utils
"""
def distance_correlation(x, y):
    """
    두 벡터 x, y에 대해 Distance Correlation 계산.

    Parameters:
        x (list or numpy array): 첫 번째 데이터 벡터.
        y (list or numpy array): 두 번째 데이터 벡터.

    Returns:
        float: Distance Correlation 값.
    """
    # 거리 행렬 계산
    def distance_matrix(a):
        n = len(a)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = abs(a[i] - a[j])
        return dist

    # 중심화 거리 행렬 계산
    def center_distance_matrix(dist):
        n = dist.shape[0]
        row_mean = np.mean(dist, axis=1).reshape(-1, 1)
        col_mean = np.mean(dist, axis=0).reshape(1, -1)
        total_mean = np.mean(dist)
        return dist - row_mean - col_mean + total_mean

    # 거리 행렬과 중심화 거리 행렬
    dist_x = distance_matrix(x)
    dist_y = distance_matrix(y)
    dist_x_centered = center_distance_matrix(dist_x)
    dist_y_centered = center_distance_matrix(dist_y)

    # Distance Covariance
    dcov_xy = np.sqrt(np.mean(dist_x_centered * dist_y_centered))
    dcov_xx = np.sqrt(np.mean(dist_x_centered * dist_x_centered))
    dcov_yy = np.sqrt(np.mean(dist_y_centered * dist_y_centered))

    # Distance Correlation 계산
    if dcov_xx * dcov_yy == 0:
        return 0  # 두 벡터가 독립적인 경우
    else:
        return dcov_xy / np.sqrt(dcov_xx * dcov_yy)
    
def bimodality_coefficient(data, data_skew, data_kurtosis):
    n = len(data)
    # g = skew(data)  # 왜도
    # k = kurtosis(data, fisher=True)  # 첨도 (Fisher’s definition)
    
    bc = (data_skew**2 + 1) / (data_kurtosis + (3 * (n - 1)**2) / ((n - 2) * (n - 3)))
    return bc