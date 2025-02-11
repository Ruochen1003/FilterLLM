'''
MIT License
Copyright (c) 2024 Yaochen Zhu
'''
import time
from tqdm import tqdm
import numpy as np
import random

def Recall_at_k(y_true, y_pred, k, agg="mean"):
    '''
        Average recall for top k recommended results.
        The training records should be set to -inf in y_pred
    '''
    batch_size = y_pred.shape[0]
    topk_idxes = np.argpartition(-y_pred, k, axis=1)[:, :k]
    y_pred_bin = np.zeros_like(y_pred, dtype=bool)
    y_pred_bin[np.arange(batch_size)[:, None], topk_idxes] = True
    y_true_bin = (y_true > 0)
    hits = np.sum(np.logical_and(y_true_bin, y_pred_bin), axis=-1).astype(np.float32)
    recalls = hits/np.minimum(k, np.sum(y_true_bin, axis=1))
    if agg == "sum":
        recall = np.sum(recalls)
    elif agg == "mean":
        recall = np.mean(recalls)
    else:
        raise NotImplementedError(f"aggregation method {agg} not defined!")
    return recall


def NDCG_at_k(y_true, y_pred, k, agg="mean"):
    '''
        Average NDCG for top k recommended results.
        The training records should be set to -inf in y_pred
    '''

    batch_size = y_pred.shape[0]
    topk_idxes_unsort = np.argpartition(-y_pred, k, axis=1)[:, :k]
    topk_value_unsort = y_pred[np.arange(batch_size)[:, None],topk_idxes_unsort]
    topk_idxes_rel = np.argsort(-topk_value_unsort, axis=1)
    topk_idxes = topk_idxes_unsort[np.arange(batch_size)[:, None], topk_idxes_rel]
    y_true_topk = y_true[np.arange(batch_size)[:, None], topk_idxes]
    y_true_bin = (y_true > 0).astype(np.float32)
    weights = 1./np.log2(np.arange(2, k + 2))
    DCG = np.sum(y_true_topk*weights, axis=-1)
    normalizer = np.array([np.sum(weights[:int(n)]) for n in np.minimum(k, np.sum(y_true_bin, axis=-1))])
    if agg == "sum":
        NDCG = np.sum(DCG/normalizer)
    elif agg == "mean":
        NDCG = np.mean(DCG/normalizer)
    return NDCG


def df_get_neighbors(input_df, obj, max_num):
    """
    Get users' neighboring items.
    return:
        nei_array - [max_num, neighbor array], use 0 to pad users which have no neighbors.
    """
    group = tuple(input_df.groupby(obj))
    keys, values = zip(*group)  # key: obj scalar, values: neighbor array

    keys = np.array(keys, dtype=np.int64)
    opp_obj = 'item' if obj == 'user' else 'user'
    values = list(map(lambda x: x[opp_obj].values, values))
    values.append(0)  # 防止成为一个 object 类型的二维 array
    values = np.array(values, dtype=object)

    nei_array = np.zeros((max_num, ), dtype=object)
    nei_array[keys] = values[:-1]
    return nei_array


def neighbor_to_neighbor(begin_array, end_array):
    """默认 id 都是从 0 开始重新编排的"""
    two_hop_nei_array = []
    for i, neighbors in tqdm(enumerate(begin_array)):
        if hasattr(neighbors, 'shape'):
            two_hop_neighbors_list = end_array[neighbors].tolist()
            two_hop_neighbors_set = set(np.concatenate(two_hop_neighbors_list, axis=0)) - {i}
            two_hop_neighbors_array = np.array(list(two_hop_neighbors_set), dtype=np.int64)
        else:
            two_hop_neighbors_array = 0
        two_hop_nei_array.append(two_hop_neighbors_array)

    return np.array(two_hop_nei_array, dtype=np.object)


class Timer(object):
    def __init__(self, name=''):
        self._name = name
        self.begin_time = time.time()
        self.last_time = time.time()
        self.current_time = time.time()
        self.stage_time = 0.0
        self.run_time = 0.0

    def logging(self, message):
        """
        输出当前的运行时间信息, 包括当前日期, 时刻, 过程已运行时间, 运行信息
        参数:
            message - 运行信息
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.update()
        message = '' if message is None else message
        print("{} {} {:.0f}s {:.0f}s | {}".format(current_time,
                                                  self._name,
                                                  self.run_time,
                                                  self.stage_time,
                                                  message))

    def update(self):
        self.current_time = time.time()

        self.stage_time = self.current_time - self.last_time
        self.last_time = self.current_time
        self.run_time = self.current_time - self.begin_time
        return self


def bpr_neg_samp(uni_users, n_users, support_dict, item_array):
    """
    以 user 为单位的训练样本采样方法
    无论是 pointwise，pairwise 还是 listwise，采样都是围绕 user 进行的
    pairwise 里的负采样，也是针对一个 user 采若干负样本，只不过将负样本都与同个正样本配对
    param:
        uni_users - unique users in training data
        dict - {uid: array[items]}
        n_users - sample n users
        neg_num - n of sample pairs for a user.
        item_array - sample item in this array.

    return:
        ret_array - [uid pos_iid neg_iid] * n_records
    """
    pos_items = []
    users = np.random.choice(uni_users, size=n_users, replace=True)
    for user in users:
        # pos sampling
        pos_candidates = support_dict[user]
        # if not hasattr(pos_candidates, 'shape'):
        #     continue
        pos_item = random.choice(pos_candidates)
        pos_items.append(pos_item)

    pos_items = np.array(pos_items, dtype=np.int64).flatten()
    neg_items = np.random.choice(item_array, len(users), replace=True)
    ret = np.stack([users, pos_items, neg_items], axis=1)
    return ret


def negative_sampling(pos_user_array, pos_item_array, neg, warm_item):
    """
    以 interaction 为单位的训练样本采样方法
    Args:
        pos_user_array: users in train interactions
        pos_item_array: items in train interactions
        neg: num of negative samples
        warm_item: train item set

    Returns:
        user: concat pos users and neg ones
        item: concat pos item and neg ones
        target: scores of both pos interactions and neg ones
    """
    user_pos = pos_user_array.reshape((-1))
    if neg >= 1:
        user_neg = np.tile(pos_user_array, int(neg)).reshape((-1))
    else:
        user_neg = np.random.choice(pos_user_array, size=(int(neg * len(user_pos))), replace=True)
    user_array = np.concatenate([user_pos, user_neg], axis=0)
    item_pos = pos_item_array.reshape((-1))
    item_neg = np.random.choice(warm_item, size=user_neg.shape[0], replace=True).reshape((-1))
    item_array = np.concatenate([item_pos, item_neg], axis=0)
    target_pos = np.ones_like(item_pos)
    target_neg = np.zeros_like(item_neg)
    target_array = np.concatenate([target_pos, target_neg], axis=0)
    random_idx = np.random.permutation(user_array.shape[0])  # 生成一个打乱的 range 序列作为下标
    return user_array[random_idx], item_array[random_idx], target_array[random_idx]