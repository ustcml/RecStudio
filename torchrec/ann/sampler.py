import torch
import numpy as np
from torchrec.model import scorer
import torch.nn.functional as F
def kmeans(X, K_or_center, max_iter=300, verbose=False):
    N = X.size(0)
    if isinstance(K_or_center, int) is None:
        K = K_or_center
        C = X[torch.randperm(N)[:K]]
    else:
        K = K_or_center.size(0)
        C = K_or_center
    prev_loss = np.inf
    for iter in range(max_iter):
        dist = torch.sum(X * X, dim=-1, keepdim=True) - 2 * (X @ C.T) + torch.sum(C * C, dim=-1).unsqueeze(0)
        assign = dist.argmin(-1)
        assign_m = torch.zeros(N, K)
        assign_m[(range(N), assign)] = 1
        loss = torch.sum(torch.square(X - C[assign,:])).item()
        if verbose:
            print(f'step:{iter:<3d}, loss:{loss:.3f}')
        if (prev_loss - loss) < prev_loss * 1e-6:
            break
        prev_loss = loss
        cluster_count = assign_m.sum(0)
        C =  (assign_m.T @ X) / cluster_count.unsqueeze(-1)
        empty_idx = cluster_count<.5
        ndead = empty_idx.sum().item()
        C[empty_idx] = X[torch.randperm(N)[:ndead]]
    return C, assign, assign_m, loss



def construct_index(cd01, K):
    cd01, indices = torch.sort(cd01, stable=True)
    cluster, count = torch.unique_consecutive(cd01, return_counts=True)
    count_all = torch.zeros(K**2 + 1, dtype=torch.long, device=cd01.device)
    count_all[cluster + 1] = count
    indptr = count_all.cumsum(dim=-1)
    return indices, indptr

class Sampler(torch.nn.Module):
    def __init__(self, num_items, scorer_fn=None):
        super(Sampler, self).__init__()
        self.num_items = num_items
        self.scorer = scorer_fn
    
    def update(self, item_embs, max_iter=30):
        pass

    def compute_item_p(self, query, pos_items):
        pass

class UniformSampler(Sampler):
    """
    For each user, sample negative items
    """
    def forward(self, query, num_neg, pos_items=None):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L || assume padding=0
        # not deal with reject sampling
        with torch.no_grad():
            num_queries = np.prod(query.shape[:-1])
            neg_items = torch.randint(1, self.num_items + 1, size=(num_queries, num_neg), device=query.device) # padding with zero
            neg_items = neg_items.reshape(*query.shape[:-1], -1) # B x L x Neg || B x Neg
            neg_prob = self.compute_item_p(query, neg_items)
            pos_prob = self.compute_item_p(query, pos_items)
        return pos_prob, neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return - torch.log(torch.ones_like(pos_items))
    


class PopularSamplerModel(Sampler):
    def __init__(self, pop_count, scorer=None, mode=0):
        super(PopularSamplerModel, self).__init__(pop_count.shape[0], scorer)
        with torch.no_grad():
            pop_count = torch.tensor(pop_count, dtype=torch.float)
            if mode == 0:
                pop_count = torch.log(pop_count + 1)
            elif mode == 1:
                pop_count = torch.log(pop_count + 1) + 1e-6
            elif mode == 2:
                pop_count = pop_count**0.75

            #pop_count = torch.cat([torch.zeros(1), pop_count]) ## adding a padding value
            pop_count = torch.cat([torch.ones(1), pop_count], dim=0) ## should include 1 not 0, to avoid the problem of log 0 !!!
            self.pop_prob = pop_count / pop_count.sum()
            self.table = torch.cumsum(self.pop_prob)
            self.pop_count = pop_count

    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad():
            num_queries = np.prod(query.shape[:-1])
            seeds = torch.rand(num_queries, num_neg)
            # @todo replace searchsorted with torch.bucketize
            neg_items = torch.searchsorted(self.table, seeds)
            neg_items = neg_items.reshape(query.shape[:-1], -1) # B x L x Neg || B x Neg
            neg_prob = self.compute_item_p(query, neg_items)
            pos_prob = self.compute_item_p(query, pos_items)
        return pos_prob, neg_items, neg_prob
    
    def compute_item_p(self, query, pos_items):
        return torch.log(self.pop_prob[pos_items])  # padding value with log(0)



class SoftmaxApprSamplerUniform(Sampler):
    """
    Uniform sampling for the final items
    """

    def __init__(self, num_items, num_clusters, scorer_fn=None):
        assert scorer_fn is None or not isinstance(scorer_fn, scorer.MLPScorer)
        super(SoftmaxApprSamplerUniform, self).__init__(num_items, scorer_fn)
        self.K = num_clusters

    def update(self, item_embs, max_iter=30):
        if isinstance(self.scorer, scorer.CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        #embs1, embs2 = np.array_split(item_embs, 2, axis=-1)
        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        self.c0, cd0, cd0m, _ = kmeans(embs1, self.c0 if hasattr(self, 'c0') else self.K, max_iter)
        self.c1, cd1, cd1m, _ = kmeans(embs2, self.c1 if hasattr(self, 'c1') else self.K, max_iter)
        #cluster_kmeans_0 = KMeans(n_clusters=self.K, random_state=0).fit(embs1)
        #self.c0 = torch.from_numpy(cluster_kmeans_0.cluster_centers_.T)
        #cd0 = cluster_kmeans_0.labels_
        #cluster_kmeans_1 = KMeans(n_clusters=self.K, random_state=0).fit(embs2)
        #self.c1 = torch.from_numpy(cluster_kmeans_1.cluster_centers_.T)
        #cd1 = cluster_kmeans_1.labels_
        self.c0_ = torch.cat([torch.zeros(1, self.c0.size(1)), self.c0], dim=0) ## for retreival probability, considering padding
        self.c1_ = torch.cat([torch.zeros(1, self.c1.size(1)), self.c1], dim=0) ## for retreival probability, considering padding
        self.cd0 = torch.cat([torch.tensor([-1]), cd0], dim=0) + 1 ## for retreival probability, considering padding
        self.cd1 = torch.cat([torch.tensor([-1]), cd1], dim=0) + 1 ## for retreival probability, considering padding
        cd01 = cd0 * self.K + cd1
        #self.member = sps.csc_matrix((np.ones_like(cd01), (np.arange(self.num_items), cd01)), \
        #    shape=(self.num_items, self.K**2))
        #self.indices = torch.from_numpy(self.member.indices)
        #self.indptr = torch.from_numpy(self.member.indptr)
        self.indices, self.indptr = construct_index(cd01, self.K)
        #self.wkk = torch.from_numpy(np.sum(self.member, axis=0).A).reshape(self.K, self.K)
        #return cd0m, cd1m
        self._update(item_embs, cd0m, cd1m)

    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            self.wkk = cd0m.T @ cd1m 
        else:
            norm = torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
            self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
            self.p = torch.cat([torch.tensor(1.0), norm], dim=0) # this is similar, to avoid log 0 !!! in case of zero padding 
            self.cp = norm[self.indices]
            for c in range(self.K**2):
                start, end = self.indptr[c], self.indptr[c+1]
                if end > start:
                    cumsum = self.cp[start:end].cumsum()
                    self.cp[start:end] = cumsum / cumsum[-1]

    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad:
            if isinstance(self.scorer, scorer.CosineScorer):
                query = F.normalize(query, dim=-1)
            q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
            r1 = q1 @ self.c1.T
            r1s = torch.softmax(r1, dim=-1) # num_q x K1
            r0 = q0 @ self.c0.T
            r0s = torch.softmax(r0, dim=-1) # num_q x K0
            s0 = (r1s @ self.wkk.T) * r0s # num_q x K0 | wkk: K0 x K1
            k0 = torch.multinomial(s0, num_neg, replacement=True) # num_q x neg
            p0 = torch.gather(r0, -1, k0)     # num_q * neg
            subwkk = self.wkk[k0, :]          # num_q x neg x K1
            s1 = subwkk * r1s.unsqueeze(1)     # num_q x neg x K1
            k1 = torch.multinomial(s1.view(-1, s1.size(-1)), 1).squeeze(-1).view(*s1.shape[:-1]) # num_q x neg
            p1 = torch.gather(r1, -1, k1) # num_q x neg
            k01 = k0 * self.K + k1  # num_q x neg
            p01 = p0 + p1
            neg_items, neg_prob = self.sample_item(k01, p01)
            pos_prop = self.compute_item_p(query, pos_items)
            return pos_prop, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def sample_item(self, k01, p01):
        if not hasattr(self, 'cp'):
            item_cnt = self.indptr[k01 + 1] - self.indptr[k01] # num_q x neg, the number of items
            item_idx = torch.int32(torch.floor(item_cnt * torch.rand_like(item_cnt))) # num_q x neg
            neg_items = self.indices[item_idx  + self.indptr[k01]] + 1
            neg_prob = p01
            return neg_items, neg_prob
        else:
            return self._sample_item_with_pop(k01, p01)
    
    def _sample_item_with_pop(self, k01, p01):
        # k01 num_q x neg, p01 num_q x neg
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        fullrange = start.unsqueeze(-1) + torch.arange(maxlen).reshape(1, 1, maxlen) # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        # @todo replace searchsorted with torch.bucketize
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(start).unsqueeze(-1)).squeeze(-1) ## num_q x neg
        item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]]
        neg_probs = self.p[item_idx + self.indptr[k01] + 1] # plus 1 due to considering padding, since p include num_items + 1 entries
        return  neg_items, p01 + np.log(neg_probs)
    
    def compute_item_p(self, query, pos_items):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L1 || assume padding=0
        k0 = self.cd0[pos_items] # B x L || B x L1
        k1 = self.cd1[pos_items] # B x L || B x L1
        c0 = self.c0_[k0, :] # B x L x D || B x L1 x D
        c1 = self.c1_[k1, :] # B x L x D || B x L1 x D
        q0, q1 = query.chunk(2, dim=-1) # B x L x D || B x D
        if query.dim() == pos_items.dim():
            r = torch.bmm(c0, q0.unsqueeze(-1)) + torch.bmm(c1, q1.unsqueeze(-1)).squeeze(-1) # B x L1
        else:
            r = torch.sum(c0 * q0, dim=-1) + torch.sum(c1 * q1, dim=-1) # B x L
        if not hasattr(self, 'p'):
            return r
        else:
            return r + np.log(self.p[pos_items])

class SoftmaxApprSamplerPop(SoftmaxApprSamplerUniform):
    """
    Popularity sampling for the final items
    """
    def __init__(self, pop_count, num_cluster, scorer=None, mode=1):
        super(SoftmaxApprSamplerPop, self).__init__(pop_count.shape[0], num_cluster, scorer)
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_count = torch.from_numpy(pop_count)
    
    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            norm = self.pop_count
        else:
            norm = self.pop_count * torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
        self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
        #self.p = torch.from_numpy(np.insert(pop_count, 0, 1.0)) # this is similar, to avoid log 0 !!! in case of zero padding 
        self.p = torch.cat([torch.tensor(1.0), norm], dim=0) # this is similar, to avoid log 0 !!! in case of zero padding 
        self.cp = norm[self.indices]
        for c in range(self.K**2):
            start, end = self.indptr[c], self.indptr[c+1]
            if end > start:
                cumsum = self.cp[start:end].cumsum()
                self.cp[start:end] = cumsum / cumsum[-1]
        


    # def sample_item(self, k01, p01):
    #     # k01 num_q x neg, p01 num_q x neg
    #     start = self.indptr[k01]
    #     last = self.indptr[k01 + 1] - 1
    #     count = last - start + 1
    #     maxlen = count.max()
    #     fullrange = start.unsqueeze(-1) + torch.arange(maxlen).reshape(1, 1, maxlen) # num_q x neg x maxlen
    #     fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
    #     # @todo replace searchsorted with torch.bucketize
    #     item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(start).unsqueeze(-1)).squeeze(-1) ## num_q x neg
    #     item_idx = torch.minimum(item_idx, last)
    #     neg_items = self.indices[item_idx + self.indptr[k01]]
    #     neg_probs = self.p[item_idx + self.indptr[k01] + 1] # plus 1 due to considering padding, since p include num_items + 1 entries
    #     return  neg_items, p01 + np.log(neg_probs)
    
    # def compute_item_p(self, query, pos_items):
    #     r = super().compute_item_p(query, pos_items)
    #     p_r = self.p[pos_items]
    #     return r + np.log(p_r)

#TODO add clustering based sampler 
#TODO aobpr sampler