# this provides some basic utilities, such as matrix split, read file into a matrix

import scipy as sp
import scipy.sparse as ss
import scipy.io as sio
import random
import numpy as np
from typing import List
import bisect
class Eval:
    @staticmethod
    def evaluate_item(train:ss.csr_matrix, test:ss.csr_matrix, user:np.ndarray, item:np.ndarray, topk:int=200, cutoff:int=200):
        train = train.tocsr()
        test = test.tocsr()
        idx = np.squeeze((test.sum(axis=1) > 0).A)
        train = train[idx, :]
        test = test[idx, :]
        user = user[idx, :]
        N = train.shape[1]
        cand_count = N - train.sum(axis=1)
        if topk <0:
            mat_rank = Eval.predict(train, test, user, item)
        else:
            mat_rank = Eval.topk_search_(train, test, user, item, topk)
        return Eval.compute_item_metric(test, mat_rank, cand_count, cutoff)

    @staticmethod
    def compute_item_metric(test:ss.csr_matrix, mat_rank:ss.csr_matrix, cand_count:np.ndarray, cutoff:int=200):
        rel_count = (test !=0).sum(axis=1)
        istopk = mat_rank.max() < test.shape[1] * 0.5
        recall, precision, map = Eval.compute_recall_precision(mat_rank, rel_count, cutoff)
        ndcg = Eval.compute_ndcg(test, mat_rank, cutoff)
        if not istopk:
            auc, mpr = Eval.compute_auc(mat_rank, rel_count, cand_count)
            return {'item_recall': recall, 'item_prec': precision, 'item_map': map, 'item_ndcg': ndcg, 'item_mpr':mpr, 'item_auc':auc}
        else:
            return {'item_recall':recall, 'item_prec':precision, 'item_map': map, 'item_ndcg':ndcg}

    @staticmethod
    def compute_ndcg(test, mat_rank, cutoff):
        M, _ = test.shape
        mat_rank_ = mat_rank.tocoo()
        user, item, rank = mat_rank_.row, mat_rank_.col, mat_rank_.data
        score = np.squeeze(test[(user, item)].A) / np.log2(rank + 2)
        dcg_score = ss.csr_matrix((score, (user, rank)), shape=test.shape)
        dcg = np.cumsum(dcg_score[:, :cutoff].todense(), axis=1)
        dcg = np.c_[dcg, dcg_score.sum(axis=1)]
        idcg = np.zeros((M, cutoff+1))
        for i in range(M):
            r = test.data[test.indptr[i]:test.indptr[i+1]]
            idcg_ = np.cumsum(-np.sort(-r) / np.log2(np.array(range(len(r)))+2))
            if cutoff > len(r):
                idcg[i,:] = np.r_[idcg_, np.tile(idcg_[-1], cutoff+1-len(r))]
            else:
                idcg[i,:] = np.r_[idcg_[:cutoff], idcg_[-1]]
        ndcg = dcg / idcg
        ndcg = np.mean(ndcg, axis=0)
        return np.squeeze(ndcg.A)
    
    @staticmethod
    def compute_recall_precision(mat_rank, user_count, cutoff):
        user_count = user_count.A.T
        M, _ = mat_rank.shape
        mat_rank_ = mat_rank.tocoo()
        user, rank = mat_rank_.row, mat_rank_.data
        user_rank = ss.csr_matrix((np.ones_like(user), (user, rank)), shape=mat_rank.shape)
        user_rank = user_rank[:,:cutoff].todense()
        user_count_inv = ss.diags(1/user_count, [0])
        cum = np.cumsum(user_rank, axis=1)
        recall = np.mean(user_count_inv * cum, axis=0)
        prec_cum = cum * ss.diags(1/np.array(range(1,cutoff+1)), 0)
        prec = np.mean(prec_cum, axis=0)
        div = np.minimum(np.tile(range(1, cutoff+1), (M, 1)), np.tile(user_count.T, (1, cutoff)))
        map = np.mean(np.divide(np.cumsum(np.multiply(prec_cum,user_rank), axis=1), div), axis=0)
        return np.squeeze(recall.A), np.squeeze(prec.A), np.squeeze(map.A)
    
    @staticmethod
    def compute_auc(mat_rank, rel_count, cand_count):
        rel_count = rel_count.A
        cand_count = cand_count.A
        tmp = mat_rank.sum(axis=1) 
        mpr = np.mean(tmp / cand_count / rel_count)
        auc_vec = rel_count * cand_count - tmp - rel_count - rel_count * (rel_count - 1) / 2
        auc_vec = auc_vec / ((cand_count - rel_count) * rel_count)
        auc = np.mean(auc_vec)
        return auc, mpr
    @staticmethod
    def evaluate_item_with_code(train:ss.csr_matrix, test:ss.csr_matrix, user:np.ndarray, item_code: np.ndarray, item_center: List[np.ndarray], topk=200, cutoff=200):
        train = train.tocsr()
        test = test.tocsr()
        #result1 = Eval.topk_search_with_code(train, user, item_code, item_center, topk)
        result = Eval.topk_search_with_code_fast(train, user, item_code, item_center, topk)
        return Eval.evaluate_topk(train, test, result, cutoff)
    #@staticmethod
    #def topk_search_approximate(train:ss.csr_matrix, user:np.ndarray, item_code: np.ndarray, item_center: List[np.ndarray]):
    @staticmethod
    def topk_search_with_code_fast(train:ss.csr_matrix, user:np.ndarray, item_code: np.ndarray, item_center: List[np.ndarray], topk=200):
        M, _ = train.shape
        traind = [train.indices[train.indptr[i]:train.indptr[i + 1]].tolist() for i in range(M)]
        center = np.concatenate(item_center)
        #result = uts.topk_search_with_code(traind, user, item_code, center, topk)
        #return result.reshape([M, topk])
        return None
    @staticmethod
    def topk_search_with_code(train:ss.csr_matrix, user:np.ndarray, item_code: np.ndarray, item_center: List[np.ndarray], topk=200):
        item_center = np.stack(item_center, 0)  # m x K x D
        M = train.shape[0]
        result = np.zeros((M, topk), dtype=np.int)
        for i in range(M):
            E = train.indices[train.indptr[i]:train.indptr[i + 1]]
            center_score = np.tensordot(item_center, user[i,:], [-1, -1])  # m x K
            #pred = uts.fetch_score(item_code, center_score)
            #pred[E] = -np.inf
            #idx = np.argpartition(pred, -topk)[-topk:]
            #result[i, :] = idx[np.argsort(-pred[idx])]
        return result

    @staticmethod
    def item_reranking(topk_item: np.ndarray, score_func):
        M, K = topk_item.shape
        result = np.zeros_like(topk_item)
        for i in range(M):
            score_item = [(topk_item[i, k], score_func(i, topk_item[i, k])) for k in range(K)]
            result[i,:] = [a for (a, b) in sorted(score_item, key=lambda x: -x[1])]
        return result

    @staticmethod
    def evaluate_topk(train:ss.csr_matrix, test:ss.csr_matrix, topk_item:np.ndarray, cutoff:int=200):
        train = train.tocsr()
        test = test.tocsr()
        result = topk_item
        N = train.shape[1]
        cand_count = N - train.sum(axis=1)
        M = test.shape[0]
        uir = []
        for i in range(M):
            R = set(test.indices[test.indptr[i]:test.indptr[i+1]])
            for k in range(result.shape[1]):
                if result[i,k] in R:
                    uir.append((i, result[i,k], k))
        user_id, item_id, rank = zip(*uir) 
        mat_rank = ss.csr_matrix((rank, (user_id, item_id)), shape=test.shape)
        return Eval.compute_item_metric(test, mat_rank, cand_count, cutoff)

    @staticmethod
    def topk_search(train:ss.csr_matrix, user:np.ndarray, item:np.ndarray, topk:int=200)->np.ndarray:
        train = train.tocsr()
        M, _ = train.shape
        item_t = item.T
        result = np.zeros((M, topk), dtype=np.int)
        for i in range(M):
            E = train.indices[train.indptr[i]:train.indptr[i+1]]
            pred = np.matmul(user[i,:], item_t)
            #pred = np.tensordot(user[i,:], item, [0,-1])
            pred[E] = -np.inf
            idx = np.argpartition(pred, -topk)[-topk:]
            result[i,:] = idx[np.argsort(-pred[idx])]
        return result

    @staticmethod
    def topk_search_(train:ss.csr_matrix, test:ss.csr_matrix, user:np.ndarray, item:np.ndarray, topk:int=200)->ss.csr_matrix:
        M, _ = train.shape
        #traind = [train.indices[train.indptr[i]:train.indptr[i + 1]].tolist() for i in range(M)]
        #result = uts.topk_search(traind, user, item, topk).reshape([M, topk])
        result = Eval.topk_search(train, user, item, topk)
        uir = []
        for i in range(M):
            R = set(test.indices[test.indptr[i]:test.indptr[i+1]])
            for k in range(topk):
                if result[i,k] in R:
                    uir.append((i, result[i,k], k))
        user_id, item_id, rank = zip(*uir) 
        mat_rank = ss.csr_matrix((rank, (user_id, item_id)), shape=test.shape)
        return mat_rank        
        #user_id, rank = result.nonzero()
        #item_id = result[(user_id, rank)]
        #mat_rank = sp.csr_matrix((rank, (user_id, item_id)), shape=test.shape)
        #return mat_rank.multiply(test !=0)

    @staticmethod
    def predict(train:ss.csr_matrix, test:ss.csr_matrix, user:np.ndarray, item:np.ndarray)->ss.csr_matrix:
        M, _ = train.shape
        item_t = item.T
        full_rank = np.zeros_like(test.data)
        for i in range(M):
            E = train.indices[train.indptr[i]:train.indptr[i+1]]
            R = test.indices[test.indptr[i]:test.indptr[i+1]]
            U = user[i,:]
            pred = np.matmul(U, item_t)
            pred[E] = -np.inf
            idx = np.argsort(-pred)
            rank = np.zeros_like(idx)
            rank[idx] = range(len(idx))
            full_rank[test.indptr[i]:test.indptr[i+1]] = rank[R]
        mat_rank = ss.csr_matrix((full_rank, test.indices, test.indptr), shape=test.shape)
        return mat_rank
    
    @staticmethod
    def format(metric:dict):
        list_str = []
        for k, v in metric.items():
            if 'ndcg' in k:
                m_str = '{0:11}:[{1}, {2:.4f}]'.format(k ,', '.join('{:.4f}'.format(e) for e in v[(10-1)::10]), v[-1])
            elif not isinstance(v, np.ndarray):
                m_str = '{0:11}:{1:.4f}'.format(k , v)
            else:
                m_str = '{0:11}:[{1}]'.format(k ,', '.join('{:.4f}'.format(e) for e in v[(10-1)::10]))
            list_str.append(m_str)
        return '\n'.join(list_str)
    
    #@staticmethod
    #def mean_and_merge(metrics: list(dict):


class IO:
    @staticmethod
    def load_matrix_from_file(filename=''):
        if filename.endswith('.mat'):
            return sio.loadmat(filename)['data']
        elif filename.endswith('.txt') or filename.endswith('.tsv'):
            sep = '\t'
        elif filename.endswith('.csv'):
            sep = ','
        else:
            raise ValueError('not supported file type')
        max_user = -1
        max_item = -1
        row_idx = []
        col_idx = []
        data = []
        for line in open(filename):
            user, item, rating = line.strip().split(sep)
            user, item, rating = int(user), int(item), float(rating)
            row_idx.append(user)
            col_idx.append(item)
            data.append(rating)
            if user > max_user:
                max_user = user
            if item > max_item:
                max_item = item
        return sp.sparse.csc_matrix((data, (row_idx, col_idx)), (max_user+1, max_item+1))
    
    @staticmethod
    def save_matrix(filename, mat, sep='\t'):
        newmat = mat.tocoo()
        with open(filename, 'w+') as f:
            for (u,i,r) in zip(newmat.row, newmat.col, newmat.data):
                f.write('{0}{3}{1}{3}{2}\n'.format(u, i, r, sep))
    @staticmethod    
    def split_matrix(mat, ratio=0.8):
        mat = mat.tocsr()
        m,n = mat.shape
        train_data_indices = []
        train_indptr = [0] * (m+1)
        test_data_indices = []
        test_indptr = [0] * (m+1)
        for i in range(m):
            row = [(mat.indices[j], mat.data[j]) for j in range(mat.indptr[i], mat.indptr[i+1])]
            train_idx = random.sample(range(len(row)), round(ratio * len(row)))
            train_binary_idx = np.full(len(row), False)
            train_binary_idx[train_idx] = True
            test_idx = (~train_binary_idx).nonzero()[0]
            for idx in train_idx:
                train_data_indices.append(row[idx]) 
            train_indptr[i+1] = len(train_data_indices)
            for idx in test_idx:
                test_data_indices.append(row[idx])
            test_indptr[i+1] = len(test_data_indices)

        [train_indices, train_data] = zip(*train_data_indices)
        [test_indices, test_data] = zip(*test_data_indices)

        train_mat = sp.sparse.csr_matrix((train_data, train_indices, train_indptr), (m,n))
        test_mat = sp.sparse.csr_matrix((test_data, test_indices, test_indptr), (m,n))

        return train_mat, test_mat
