# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

from utils import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class POIDataset(Dataset):
    def __init__(self, data_filename, pois_coos_filename, num_users, num_pois, padding_idx, args, device):

        # get all sessions and labels
        self.data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
        self.sessions_dict = self.data[0]  # poiID starts from 0
        self.labels_dict = self.data[1]
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)

        # definition
        self.num_users = num_users
        self.num_pois = num_pois
        # self.num_sessions = get_num_sessions(self.sessions_dict)
        self.padding_idx = padding_idx
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.device = device

        # get user's trajectory, reversed trajectory and its length
        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(self.sessions_dict)
        self.users_rev_trajs_dict = get_user_reverse_traj(self.users_trajs_dict)

        # calculate poi-poi haversine distance and generate geographical adjacency matrix
        self.poi_geo_adj = gen_poi_geo_adj(num_pois, self.pois_coos_dict, self.distance_threshold)   # csr_matrix
        self.poi_geo_graph_matrix = normalized_adj(adj=self.poi_geo_adj, is_symmetric=False)
        self.poi_geo_graph = transform_csr_matrix_to_tensor(self.poi_geo_graph_matrix).to(device)

        # generate poi-session incidence matrix, its degree and hypergraph
        self.H_pu = gen_sparse_H_user(self.sessions_dict, num_pois, self.num_users)    # [L, U]
        # drop edge on csr_matrix H_pu
        self.H_pu = csr_matrix_drop_edge(self.H_pu, args.keep_rate)
        # get degree of H_pu
        self.Deg_H_pu = get_hyper_deg(self.H_pu)    # [L, L]
        # normalize poi-user hypergraph
        self.HG_pu = self.Deg_H_pu * self.H_pu    # [L, U]
        self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(device)

        # generate session-poi incidence matrix, its degree and hypergraph
        self.H_up = self.H_pu.T    # [U, L]
        self.Deg_H_up = get_hyper_deg(self.H_up)    # [U, U]
        self.HG_up = self.Deg_H_up * self.H_up    # [U, L]
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

        # get all sessions for intra-sequential relation learning
        # self.all_train_sessions = get_all_sessions(self.sessions_dict)    # list of tensor
        self.all_train_sessions = get_all_users_seqs(self.users_trajs_dict)
        # pad session into a unified fixed length
        self.pad_all_train_sessions = pad_sequence(self.all_train_sessions, batch_first=True, padding_value=padding_idx)
        self.pad_all_train_sessions = self.pad_all_train_sessions.to(device)    # [U, MAX_SEQ_LEN]
        self.max_session_len = self.pad_all_train_sessions.size(1)

        # generate directed poi-poi hypergraph
        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, num_pois)    # [L, L]
        # drop edge on csr_matrix H_pu
        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, args.keep_rate_poi)
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)    # [L, L]
        self.HG_poi_src = self.Deg_H_poi_src * self.H_poi_src    # [L, L]
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.HG_poi_src).to(device)

        # generate targeted poi hypergraph
        self.H_poi_tar = self.H_poi_src.T    # [L, L]
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)    # [L, L]
        self.HG_poi_tar = self.Deg_H_poi_tar * self.H_poi_tar    # [L, L]
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.HG_poi_tar).to(device)

    def __len__(self):
        return self.num_users

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = self.users_trajs_lens_dict[user_idx]
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = self.users_rev_trajs_dict[user_idx]
        label = self.labels_dict[user_idx]

        sample = {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }

        return sample


class POIPartialDataset(Dataset):
    def __init__(self, full_dataset, user_indices):
        self.data = [full_dataset[i] for i in user_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class POISessionDataset(Dataset):
    def __init__(self, data_filename, label_filename, pois_coos_filename, num_pois, padding_idx, args, device):

        # get all sessions and labels
        # self.data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
        # self.sessions_dict = self.data[0]  # poiID starts from 0
        # self.labels_dict = self.data[1]
        self.sessions_dict = load_dict_from_pkl(data_filename)
        self.labels_dict = load_dict_from_pkl(label_filename)
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)
        self.users_trajs_dict = self.sessions_dict

        # definition
        # self.num_users = num_users
        self.num_pois = num_pois
        self.num_sessions = len(self.sessions_dict)
        self.padding_idx = padding_idx
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.device = device

        # calculate poi-poi haversine distance and generate geographical adjacency matrix
        self.poi_geo_adj = gen_poi_geo_adj(num_pois, self.pois_coos_dict, self.distance_threshold)   # csr_matrix
        self.poi_geo_graph_matrix = normalized_adj(adj=self.poi_geo_adj, is_symmetric=False)
        self.poi_geo_graph = transform_csr_matrix_to_tensor(self.poi_geo_graph_matrix).to(device)

        # generate directed poi-poi hypergraph
        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, num_pois)    # [L, L]
        # drop edge on csr_matrix H_pu
        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, args.keep_rate_poi)
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)    # [L, L]
        self.HG_poi_src = self.Deg_H_poi_src * self.H_poi_src    # [L, L]
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.HG_poi_src).to(device)

        # generate targeted poi hypergraph
        self.H_poi_tar = self.H_poi_src.T    # [L, L]
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)    # [L, L]
        self.HG_poi_tar = self.Deg_H_poi_tar * self.H_poi_tar    # [L, L]
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.HG_poi_tar).to(device)

        # poi-session collaborative hypergraph
        self.H_poi_session = gen_sparse_H_pois_session(self.sessions_dict, num_pois, self.num_sessions)
        self.HG_col = gen_HG_from_sparse_H(self.H_poi_session)
        self.HG_col = transform_csr_matrix_to_tensor(self.HG_col).to(device)

        # generate poi-session incidence matrix, its degree and hypergraph
        # self.H_pu = gen_sparse_H_user(self.sessions_dict, num_pois, self.num_users)  # [L, U]
        self.H_pu = self.H_poi_session
        # drop edge on csr_matrix H_pu
        # self.H_pu = csr_matrix_drop_edge(self.H_pu, args.keep_rate)

        # get degree of H_pu
        # self.Deg_H_pu = get_hyper_deg(self.H_pu)  # [L, L]
        # normalize poi-user hypergraph
        # self.HG_pu = self.Deg_H_pu * self.H_pu  # [L, U]
        # self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(device)

        # generate session-poi incidence matrix, its degree and hypergraph
        self.H_up = self.H_pu.T  # [U, L]
        self.Deg_H_up = get_hyper_deg(self.H_up)  # [U, U]
        self.HG_up = self.Deg_H_up * self.H_up  # [U, L]
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

    def __len__(self):
        # return self.num_users
        return self.num_sessions

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = len(user_seq)
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = user_seq[::-1]
        label = self.labels_dict[user_idx]

        sample = {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }

        return sample


def collate_fn_4sq(batch, padding_value=3835):
    """
    Pad sequence in the batch into a fixed length
    batch: list obj
    """
    # get each item in the batch
    batch_user_idx = []
    batch_user_seq = []
    batch_user_rev_seq = []
    batch_user_seq_len = []
    batch_user_seq_mask = []
    batch_label = []
    for item in batch:
        batch_user_idx.append(item["user_idx"])
        batch_user_seq_len.append(item["user_seq_len"])
        batch_label.append(item["label"])
        batch_user_seq.append(item["user_seq"])
        batch_user_rev_seq.append(item["user_rev_seq"])
        batch_user_seq_mask.append(item["user_seq_mask"])

    # pad seq and rev seq
    pad_user_seq = pad_sequence(batch_user_seq, batch_first=True, padding_value=padding_value)
    pad_user_rev_seq = pad_sequence(batch_user_rev_seq, batch_first=True, padding_value=padding_value)
    pad_user_seq_mask = pad_sequence(batch_user_seq_mask, batch_first=True, padding_value=0)

    # stack list obj to a torch.tensor
    batch_user_idx = torch.stack(batch_user_idx)
    batch_user_seq_len = torch.stack(batch_user_seq_len)
    batch_label = torch.stack(batch_label)

    collate_sample = {
        "user_idx": batch_user_idx,
        "user_seq": pad_user_seq,
        "user_rev_seq": pad_user_rev_seq,
        "user_seq_len": batch_user_seq_len,
        "user_seq_mask": pad_user_seq_mask,
        "label": batch_label,
    }

    return collate_sample

