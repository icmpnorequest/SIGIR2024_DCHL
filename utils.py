# coding=utf-8
"""
@author: Yantong Lai
"""

import pickle
import numpy as np
from math import radians, cos, sin, asin, sqrt
import scipy.sparse as sp
import torch


def get_unique_seq(sessions_list):
    """Get unique POIs in the sequence"""
    seq_list = []
    for session in sessions_list:
        for poi in session:
            if poi in seq_list:
                continue
            else:
                seq_list.append(poi)

    return seq_list


def get_unique_seqs_for_sessions(sessions_dict):
    """Get unique seq for each session"""
    seqs_dict = {}
    seqs_lens_dict = {}
    for key, value in sessions_dict.items():
        seqs_dict[key] = get_unique_seq(value)
        seqs_lens_dict[key] = len(get_unique_seq(value))

    return seqs_dict, seqs_lens_dict


def get_seqs_for_sessions(sessions_dict, padding_idx, max_seq_len):
    seqs_dict = {}
    seqs_lens_dict = {}
    reverse_seqs_dict = {}
    for key, sessions in sessions_dict.items():
        temp = []
        for session in sessions:
            temp.extend(session)
        if len(temp) >= max_seq_len:
            temp = temp[-max_seq_len:]
            temp_rev = temp[::-1]
            seqs_dict[key] = temp
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = max_seq_len
        else:
            temp_new = temp + [padding_idx] * (max_seq_len - len(temp))
            temp_rev = temp[::-1] + [padding_idx] * (max_seq_len - len(temp))
            seqs_dict[key] = temp_new
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = len(temp)

    return seqs_dict, reverse_seqs_dict, seqs_lens_dict


def save_list_with_pkl(filename, list_obj):
    with open(filename, 'wb') as f:
        pickle.dump(list_obj, f)


def load_list_with_pkl(filename):
    with open(filename, 'rb') as f:
        list_obj = pickle.load(f)

    return list_obj


def save_dict_to_pkl(pkl_filename, dict_pbj):
    with open(pkl_filename, 'wb') as f:
        pickle.dump(dict_pbj, f)


def load_dict_from_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        dict_obj = pickle.load(f)

    return dict_obj


def get_num_sessions(sessions_dict):
    num_sessions = 0
    for value in sessions_dict.values():
        num_sessions += len(value)

    return num_sessions


def get_user_complete_traj(sessions_dict):
    """Get each user's complete trajectory from her sessions"""
    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for userID, sessions in sessions_dict.items():
        traj = []
        for session in sessions:
            traj.extend(session)
        users_trajs_dict[userID] = traj
        users_trajs_lens_dict[userID] = len(traj)

    return users_trajs_dict, users_trajs_lens_dict


def get_user_reverse_traj(users_trajs_dict):
    """Get each user's reversed trajectory according to her complete trajectory"""
    users_rev_trajs_dict = {}
    for userID, traj in users_trajs_dict.items():
        rev_traj = traj[::-1]
        users_rev_trajs_dict[userID] = rev_traj

    return users_rev_trajs_dict


def gen_poi_geo_adj(num_pois, pois_coos_dict, distance_threshold):
    """Generate geogrpahical adjacency matrix with pois_coos_dict and distance_threshold"""
    poi_geo_adj = np.zeros(shape=(num_pois, num_pois))

    # traverse
    for poi1 in range(num_pois):
        lat1, lon1 = pois_coos_dict[poi1]
        for poi2 in range(poi1, num_pois):
            lat2, lon2 = pois_coos_dict[poi2]
            hav_dist = haversine_distance(lon1, lat1, lon2, lat2)
            if hav_dist <= distance_threshold:
                poi_geo_adj[poi1, poi2] = 1
                poi_geo_adj[poi2, poi1] = 1

    # transform np.ndarray to csr_matrix
    poi_geo_adj = sp.csr_matrix(poi_geo_adj)

    return poi_geo_adj


def process_users_seqs(users_seqs_dict, padding_idx, max_seq_len):
    processed_seqs_dict = {}
    reverse_seqs_dict = {}
    for key, seq in users_seqs_dict.items():
        if len(seq) >= max_seq_len:
            temp_seq = seq[-max_seq_len:]
            temp_rev_seq = temp_seq[::-1]
        else:
            temp_seq = seq + [padding_idx] * (max_seq_len - len(seq))
            temp_rev_seq = seq[::-1] + [padding_idx] * (max_seq_len - len(seq))
        processed_seqs_dict[key] = temp_seq
        reverse_seqs_dict[key] = temp_rev_seq

    return processed_seqs_dict, reverse_seqs_dict


def reverse_users_seqs(processed_users_seqs_dict, padding_idx, max_seq_len):
    reversed_users_seqs_dict = {}
    for key, seq in processed_users_seqs_dict.items():
        for idx in range(len(seq)):
            if seq[idx] == padding_idx:
                actual_seq = seq[:idx]
                reversed_users_seqs_dict[key] = actual_seq[::-1] + [padding_idx] * (max_seq_len - idx)
                break

    return reversed_users_seqs_dict


def gen_users_seqs_masks(users_seqs_dict, padding_idx):
    users_seqs_masks_dict = {}
    for key, seq in users_seqs_dict.items():
        temp_seq = []
        for poi in seq:
            if poi != padding_idx:
                temp_seq.append(1)
            else:
                temp_seq.append(0)
        users_seqs_masks_dict[key] = temp_seq

    return users_seqs_masks_dict


def haversine_distance(lon1, lat1, lon2, lat2):
    """Haversine distance"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371

    return c * r


def euclidean_distance(lon1, lat1, lon2, lat2):
    """Euclidean distance"""

    return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


def gen_geo_seqs_adjs_dict(users_seqs_dict, pois_coos_dict, max_seq_len, padding_idx, eta=1, distance_threshold=2.5, distance_type="haversine"):
    """Generate geographical sequential adjacency dictionary"""
    geo_adjs_dict = {}
    for key, seq in users_seqs_dict.items():
        geo_adj = np.zeros(shape=(max_seq_len, max_seq_len))
        actual_seq = []
        for item in seq:
            if item != padding_idx:
                actual_seq.append(item)
        actual_seq_len = len(actual_seq)
        for i in range(actual_seq_len):
            for j in range(i + 1, actual_seq_len):
                l1 = actual_seq[i]
                l2 = actual_seq[j]
                lat1, lon1 = pois_coos_dict[l1]
                lat2, lon2 = pois_coos_dict[l2]
                if distance_type == "haversine":
                    dist = haversine_distance(lon1, lat1, lon2, lat2)
                elif distance_type == "euclidean":
                    dist = euclidean_distance(lon1, lat1, lon2, lat2)
                if 0 < dist <= distance_threshold:
                    geo_influence = np.exp(-eta * (dist ** 2))
                    geo_adj[i, j] = geo_influence
                    geo_adj[j, i] = geo_influence
        geo_adjs_dict[key] = geo_adj

    return geo_adjs_dict


def create_user_poi_adj(users_seqs_dict, num_users, num_pois):
    """Create user-POI interaction matrix"""
    R = sp.dok_matrix((num_users, num_pois), dtype=np.float)
    for userID, seq in users_seqs_dict.items():
        for itemID in seq:
            itemID = itemID - num_users
            R[userID, itemID] = 1

    return R, R.T


def gen_sparse_interaction_matrix(users_seqs_dict, num_users, num_pois):
    """Generate sparse user-POI adjacent matrix"""
    R, R_T = create_user_poi_adj(users_seqs_dict, num_users, num_pois)
    A = sp.dok_matrix((num_users + num_pois, num_users + num_pois), dtype=float)
    A[:num_users, num_users:] = R
    A[num_users:, :num_users] = R_T
    A_sparse = A.tocsr()

    return A_sparse


def normalized_adj(adj, is_symmetric=True):
    """Normalize adjacent matrix for GCN"""
    if is_symmetric:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1/2).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj * d_mat_inv
    else:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj

    return norm_adj


def normalized_adj_tensor(adj_tensor):
    """Normalized adjacent tensor"""
    # Compute the degree matrix
    degree_tensor = torch.diag(torch.sum(adj_tensor, dim=1))

    # inverse degree
    inverse_degree_tensor = torch.inverse(degree_tensor)

    # normalized adjacency
    norm_adj = torch.matmul(inverse_degree_tensor, adj_tensor)

    # convert the normalized adjacency matrix to a sparse tensor
    sparse_norm_adj = torch.sparse.FloatTensor(norm_adj)

    return sparse_norm_adj


def gen_local_graph(adj):
    """Add self loop"""
    G = normalized_adj(adj + sp.eye(adj.shape[0]))

    return G


def gen_sparse_H(sessions_dict, num_pois, num_sessions, start_poiID):
    """Generate sparse incidence matrix for hypergraph"""
    H = np.zeros(shape=(num_pois, num_sessions))
    sess_idx = 0
    for key, sessions in sessions_dict.items():
        for session in sessions:
            for poiID in session:
                new_poiID = poiID - start_poiID
                H[new_poiID, sess_idx] = 1
            sess_idx += 1
    assert sess_idx == num_sessions
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_pois_session(sessions_dict, num_pois, num_sessions):
    H = np.zeros(shape=(num_pois, num_sessions))
    for sess_idx, session in sessions_dict.items():
        for poi in session:
            H[poi, sess_idx] = 1
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_user(sessions_dict, num_pois, num_users):
    """Generate sparse incidence matrix for hypergraph"""
    H = np.zeros(shape=(num_pois, num_users))

    for userID, sessions in sessions_dict.items():
        seq = []
        for session in sessions:
            seq.extend(session)
        for poi in seq:
            H[poi, userID] = 1

    H = sp.csr_matrix(H)

    return H


def gen_sparse_directed_H_poi(users_trajs_dict, num_pois):
    """
    Generate directed poi-poi incidence matrix for hypergraph
    Rows: source POIs
    Columns: target POIs
    """
    H = np.zeros(shape=(num_pois, num_pois))
    for userID, traj in users_trajs_dict.items():
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                H[src_poi, tar_poi] = 1
    H = sp.csr_matrix(H)

    return H


def gen_HG_from_sparse_H(H, conv="sym"):
    """Generate hypergraph with sparse incidence matrix"""
    n_edge = H.shape[1]
    W = sp.eye(n_edge)

    HW = H.dot(W)
    DV = sp.csr_matrix(HW.sum(axis=1)).astype(float)
    DE = sp.csr_matrix(H.sum(axis=0)).astype(float)
    invDE1 = DE.power(-1)
    invDE1_ = sp.diags(invDE1.toarray()[0])
    HT = H.T

    if conv == "sym":
        invDV2 = DV.power(n=-1 / 2)
        invDV2_ = sp.diags(invDV2.toarray()[:, 0])
        HG = invDV2_ * H * W * invDE1_ * HT * invDV2_
    elif conv == "asym":
        invDV1 = DV.power(-1)
        invDV1_ = sp.diags(invDV1.toarray()[:, 0])
        HG = invDV1_ * H * W * invDE1_ * HT

        # print("invDV1: \n{}".format(invDV1_.toarray()))
        # print("invDE1: \n{}".format(invDE1_.toarray()))
        # print("B-1 * HT: \n{}".format((invDE1_ * HT).toarray()))
        # print("D * H: \n{}".format((invDV1_ * H).toarray()))

    return HG


def get_hyper_deg(incidence_matrix):
    '''
    # incidence_matrix = [num_nodes, num_hyperedges]
    hyper_deg = np.array(incidence_matrix.sum(axis=axis)).squeeze()
    hyper_deg[hyper_deg == 0.] = 1
    hyper_deg = sp.diags(1.0 / hyper_deg)
    '''

    # H  = [num_node, num_edge]
    # DV = [num_node, num_node]
    # DV * H = [num_node, num_edge]

    # HT = [num_edge, num_node]
    # DE = [num_edge, num_edge]
    # DE * HT = [num_edge, num_node]

    # hyper_deg = incidence_matrix.sum(1)
    # inv_hyper_deg = hyper_deg.power(-1)
    # inv_hyper_deg_diag = sp.diags(inv_hyper_deg.toarray()[0])

    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    return d_mat_inv


def transform_csr_matrix_to_tensor(csr_matrix):
    """Transform csr matrix to tensor"""
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return sp_tensor


def get_poi_session_freq(num_pois, num_sessions, sessions_dict):
    """Get frequency occurrence of pois in sessions"""
    poi_sess_freq_matrix = np.zeros(shape=(num_pois, num_sessions))

    # traverse
    sess_idx = 0
    for userID, sessions in sessions_dict.items():
        for session in sessions:
            for poiID in session:
                poi_sess_freq_matrix[poiID, sess_idx] += 1
            sess_idx += 1

    # transform to csr_matrix
    poi_sess_freq_matrix = sp.csr_matrix(poi_sess_freq_matrix)

    return poi_sess_freq_matrix


def get_all_sessions(sessions_dict):
    """Get all sessions in the dataset"""
    all_sessions = []

    for userID, sessions in sessions_dict.items():
        for session in sessions:
            all_sessions.append(torch.tensor(session))

    return all_sessions


def get_all_users_seqs(users_trajs_dict):
    """Get all users' sequences"""
    all_seqs = []
    for userID, traj in users_trajs_dict.items():
        all_seqs.append(torch.tensor(traj))

    return all_seqs


def sparse_adj_tensor_drop_edge(sp_adj, keep_rate):
    """Drop edge on sparse adjacency tensor"""
    if keep_rate == 1.0:
        return sp_adj

    vals = sp_adj._values()
    idxs = sp_adj._indices()
    edgeNum = vals.size()
    mask = ((torch.rand(edgeNum) + keep_rate).floor()).type(torch.bool)
    newVals = vals[mask] / keep_rate
    newIdxs = idxs[:, mask]

    return torch.sparse.FloatTensor(newIdxs, newVals, sp_adj.shape)


def csr_matrix_drop_edge(csr_adj_matrix, keep_rate):
    """Drop edge on scipy.sparse.csr_matrix"""
    if keep_rate == 1.0:
        return csr_adj_matrix

    coo = csr_adj_matrix.tocoo()
    row = coo.row
    col = coo.col
    edgeNum = row.shape[0]

    # generate edge mask
    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(np.bool_)

    # get new values and indices
    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]
    new_values = np.ones(new_edgeNum, dtype=np.float)

    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    return drop_adj_matrix


