
from recstudio.model.mf.ease import EASE
import scipy.sparse as sp
import numpy as np
import torch
class ItemKNN(EASE):
    def training_epoch(self, nepoch):
        data, iscombine = self.current_epoch_trainloaders(nepoch)
        R = data['user_item_matrix']
        item_norm = np.sqrt(R.multiply(R).sum(0).A.ravel())
        item_nz = (R > 0).sum(0).A.ravel()
        G = R.T @ R
        diagIndices = np.diag_indices_from(G)
        G[diagIndices] = 0
        G.eliminate_zeros()
        all_col = []
        all_row = []
        all_val = []
        for col in range(G.shape[0]):
            if G.indptr[col] < G.indptr[col+1]:
                score = G.data[G.indptr[col]:G.indptr[col+1]]
                rows = G.indices[G.indptr[col]:G.indptr[col+1]]
                if self.config['similarity'] == 'cosine':
                    score = score / (item_norm[rows] * item_norm[col] + 1e-6)
                elif self.config['similarity'] == 'jaccard':
                    score = score / (item_nz[rows] + item_nz[col] - score + 1e-6)
                else:
                    raise ValueError('unsupported similarity metric')
                topk = self.config['knn']
                if G.indptr[col] < G.indptr[col+1] - topk:
                    idx = np.argpartition(score, -topk)[-topk:]
                    rows_ = rows[idx]
                    scores_ = score[idx]
                else:
                    rows_ = rows
                    scores_ = score
                all_col.extend([col] * len(scores_))
                all_row.extend(rows_)
                all_val.extend(scores_)

        B = sp.csc_matrix((all_val, (all_row, all_col)), G.shape)
        self.item_vector = B[:, 1:]
        self.query_encoder.user = R
        return torch.tensor(0.)