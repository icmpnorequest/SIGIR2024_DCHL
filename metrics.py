# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import numpy as np


def hit_k(y_pred, y_true, k):
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        return 1
    else:
        return 0


def ndcg_k(y_pred, y_true, k):
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        position = y_pred_indices.index(y_true) + 1
        return 1 / np.log2(1 + position)
    else:
        return 0


def mAP_metric(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    return rlt / len(y_true_seq)


def batch_performance(batch_y_pred, batch_y_true, k):
    batch_size = batch_y_pred.size(0)
    batch_recall = 0
    batch_ndcg = 0
    for idx in range(batch_size):
        hit = hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_recall += hit
        ndcg = ndcg_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_ndcg += ndcg

    recall = batch_recall / batch_size
    ndcg = batch_ndcg / batch_size

    return recall, ndcg


