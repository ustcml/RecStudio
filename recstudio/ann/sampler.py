from codecs import replace_errors
import numpy as np
import torch
import torch.nn.functional as F
from recstudio.model import scorer


def kmeans(X, K_or_center, max_iter=300, verbose=False):
    N = X.size(0)
    if isinstance(K_or_center, int):
        K = K_or_center
        C = X[torch.randperm(N)[:K]]
    else:
        K = K_or_center.size(0)
        C = K_or_center
    prev_loss = np.inf
    for iter in range(max_iter):
        dist = torch.sum(X * X, dim=-1, keepdim=True) - 2 * \
            (X @ C.T) + torch.sum(C * C, dim=-1).unsqueeze(0)
        assign = dist.argmin(-1)
        assign_m = X.new_zeros(N, K)
        assign_m[(range(N), assign)] = 1
        loss = torch.sum(torch.square(X - C[assign, :])).item()
        if verbose:
            print(f'step:{iter:<3d}, loss:{loss:.3f}')
        if (prev_loss - loss) < prev_loss * 1e-6:
            break
        prev_loss = loss
        cluster_count = assign_m.sum(0)
        C = (assign_m.T @ X) / cluster_count.unsqueeze(-1)
        empty_idx = cluster_count < .5
        ndead = empty_idx.sum().item()
        C[empty_idx] = X[torch.randperm(N)[:ndead]]
    return C, assign, assign_m, loss


def construct_index(cd01, K):
    cd01, indices = torch.sort(cd01, stable=True)
    # _, itemid2indice = torch.sort(indices, stable=True)
    cluster, count = torch.unique_consecutive(cd01, return_counts=True)
    count_all = torch.zeros(K + 1, dtype=torch.long, device=cd01.device)
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


class RetriverSampler(Sampler):
    def __init__(self, retriever, method='brute', excluding_hist=False, t=1):
        super().__init__(None)
        self.retriever = retriever
        if not method in {"ips", "dns", "brute"}:
            raise ValueError("the sampler method start with 'retriever_' only \
                support 'retriever_ipts' and 'retriever_dns'")
        else:
            self.method = method

        self.excluding_hist = excluding_hist
        self.T = t

    def _update(self):
        if hasattr(self.retriever, '_update_item_vector'): # TODO: config frequency
            self.retriever._update_item_vector()

    @torch.no_grad()
    def forward(self, batch, num_neg, pos_items, user_hist=None):
        log_pos_prob, neg_id, log_neg_prob = self.retriever.sampling(
            batch, num_neg, pos_items, user_hist, self.method, self.excluding_hist, self.T
        )
        return log_pos_prob.detach(), neg_id.detach(), log_neg_prob.detach()



class UniformSampler(Sampler):
    """
    For each user, sample negative items
    """

    def forward(self, query, num_neg, pos_items=None):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L || assume padding=0
        # not deal with reject sampling
        with torch.no_grad():
            num_queries = np.prod(query.shape[:-1])
            neg_items = torch.randint(
                1, self.num_items + 1, size=(num_queries, num_neg), device=query.device)  # padding with zero
            neg_items = neg_items.reshape(
                *query.shape[:-1], -1)  # B x L x Neg || B x Neg
            neg_prob = self.compute_item_p(query, neg_items)
            pos_prob = self.compute_item_p(query, pos_items)
        return pos_prob, neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return - torch.log(torch.ones_like(pos_items))


def uniform_sample_masked_hist(num_items: int, num_neg: int, user_hist: torch.Tensor, num_query_per_user: int = None):
    """Sampling from ``1`` to ``num_items`` uniformly with masking items in user history.

    Args:
        num_items(int): number of total items. 
        num_neg(int): number of negative samples. 
        user_hist(torch.Tensor): items list in user interacted history. The shape are required to be ``[num_user(or batch_size),max_hist_seq_len]`` with padding item(with index ``0``).
        num_query_per_user(int, optimal): number of queries if each user. It will be ``None`` when there is only one query for one user.

    Returns:
        torch.Tensor: ``[num_user(or batch_size),num_neg]`` or ``[num_user(or batch_size),num_query_per_user,num_neg]``, negative item index. If ``num_query_per_user`` is ``None``,  the shape will be ``[num_user(or batch_size),num_neg]``.
    """
    n_q = 1 if num_query_per_user is None else num_query_per_user
    num_user, hist_len = user_hist.shape
    device = user_hist.device
    neg_float = torch.rand(num_user, n_q*num_neg, device=device)
    non_zero_count = torch.count_nonzero(user_hist, dim=-1)
    neg_items = torch.floor(
        neg_float * (num_items - non_zero_count).view(-1, 1)) + 1   # BxNeg ~ U[1,2,...]
    sorted_hist, _ = user_hist.sort(dim=-1)    # BxL
    offset = torch.arange(hist_len, device=device).repeat(num_user, 1)  # BxL
    offset = offset - (hist_len - non_zero_count).view(-1, 1)
    offset[offset < 0] = 0
    sorted_hist = sorted_hist - offset
    masked_offset = torch.searchsorted(
        sorted_hist, neg_items, right=True)  # BxNeg
    padding_nums = hist_len - non_zero_count
    neg_items += (masked_offset - padding_nums.view(-1, 1))
    if num_query_per_user is not None:
        neg_items = neg_items.reshape(num_user, num_query_per_user, num_neg)
    return neg_items


class MaskedUniformSampler(Sampler):
    """
    For each user, sample negative items
    """

    def forward(self, query, num_neg, pos_items, user_hist):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L || assume padding=0
        # return BxLxN or BxN
        with torch.no_grad():
            if query.dim() == 2:
                neg_items = uniform_sample_masked_hist(
                    num_query_per_user=None, num_items=self.num_items, 
                    num_neg=num_neg, user_hist=user_hist)
            elif query.dim() == 3:
                neg_items = uniform_sample_masked_hist(num_query_per_user=query.size(
                    1), num_items=self.num_items, num_neg=num_neg, user_hist=user_hist)
            else:
                raise ValueError(
                    "`query` need to be 2-dimensional or 3-dimensional.")
            neg_prob = self.compute_item_p(query, neg_items)
            pos_prob = self.compute_item_p(query, pos_items)
        return pos_prob, neg_items.int(), neg_prob

    def compute_item_p(self, query, pos_items):
        return - torch.log(torch.ones_like(pos_items))


class DatasetUniformSampler(Sampler):
    def forward(self, num_neg=1, user_hist=None):
        for hist in user_hist:
            for i in range(num_neg):
                neg = torch.randint()
        



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

            # pop_count = torch.cat([torch.zeros(1), pop_count]) ## adding a padding value
            # should include 1 not 0, to avoid the problem of log 0 !!!
            pop_count = torch.cat([pop_count.new_ones(1), pop_count], dim=0)
            self.pop_prob = torch.nn.Parameter(
                pop_count / pop_count.sum(), requires_grad=False)
            self.table = torch.nn.Parameter(torch.cumsum(
                self.pop_prob, dim=0), requires_grad=False)
            self.pop_count = torch.nn.Parameter(pop_count, requires_grad=False)

    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad():
            num_queries = np.prod(query.shape[:-1])
            seeds = torch.rand(num_queries, num_neg, device=query.device)
            # @todo replace searchsorted with torch.bucketize
            neg_items = torch.searchsorted(self.table, seeds)
            neg_items = neg_items.reshape(
                query.shape[:-1], -1)  # B x L x Neg || B x Neg
            neg_prob = self.compute_item_p(query, neg_items)
            pos_prob = self.compute_item_p(query, pos_items)
        return pos_prob, neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return torch.log(self.pop_prob[pos_items])  # padding value with log(0)


class MIDXSamplerUniform(Sampler):
    """
    Uniform sampling for the final items
    """

    def __init__(self, num_items, num_clusters, scorer_fn=None):
        assert scorer_fn is None or not isinstance(scorer_fn, scorer.MLPScorer)
        super(MIDXSamplerUniform, self).__init__(num_items, scorer_fn)
        self.K = num_clusters

    def update(self, item_embs, max_iter=30):
        if isinstance(self.scorer, scorer.CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        self.c0, cd0, cd0m, _ = kmeans(
            embs1, self.c0 if hasattr(self, 'c0') else self.K, max_iter)
        self.c1, cd1, cd1m, _ = kmeans(
            embs2, self.c1 if hasattr(self, 'c1') else self.K, max_iter)
        # for retreival probability, considering padding
        self.c0_ = torch.cat(
            [self.c0.new_zeros(1, self.c0.size(1)), self.c0], dim=0)
        # for retreival probability, considering padding
        self.c1_ = torch.cat(
            [self.c1.new_zeros(1, self.c1.size(1)), self.c1], dim=0)
        # for retreival probability, considering padding
        self.cd0 = torch.cat([-cd0.new_ones(1), cd0], dim=0) + 1
        # for retreival probability, considering padding
        self.cd1 = torch.cat([-cd1.new_ones(1), cd1], dim=0) + 1
        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K**2)
        self._update(item_embs, cd0m, cd1m)

    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            self.wkk = cd0m.T @ cd1m
        else:
            norm = torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
            self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
            # this is similar, to avoid log 0 !!! in case of zero padding
            self.p = torch.cat([norm.new_ones(1), norm], dim=0)
            self.cp = norm[self.indices]
            for c in range(self.K**2):
                start, end = self.indptr[c], self.indptr[c+1]
                if end > start:
                    cumsum = self.cp[start:end].cumsum(0)
                    self.cp[start:end] = cumsum / cumsum[-1]

    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad():
            if isinstance(self.scorer, scorer.CosineScorer):
                query = F.normalize(query, dim=-1)
            q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
            r1 = q1 @ self.c1.T
            r1s = torch.softmax(r1, dim=-1)  # num_q x K1
            r0 = q0 @ self.c0.T
            r0s = torch.softmax(r0, dim=-1)  # num_q x K0
            s0 = (r1s @ self.wkk.T) * r0s  # num_q x K0 | wkk: K0 x K1
            k0 = torch.multinomial(
                s0, num_neg, replacement=True)  # num_q x neg
            p0 = torch.gather(r0, -1, k0)     # num_q * neg
            subwkk = self.wkk[k0, :]          # num_q x neg x K1
            s1 = subwkk * r1s.unsqueeze(1)     # num_q x neg x K1
            k1 = torch.multinomial(
                s1.view(-1, s1.size(-1)), 1).squeeze(-1).view(*s1.shape[:-1])  # num_q x neg
            p1 = torch.gather(r1, -1, k1)  # num_q x neg
            k01 = k0 * self.K + k1  # num_q x neg
            p01 = p0 + p1
            neg_items, neg_prob = self.sample_item(k01, p01)
            pos_prob = None if pos_items is None else self.compute_item_p(
                query, pos_items)
            return pos_prob, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def sample_item(self, k01, p01, pos=None):
        # TODO: remove positive items
        if not hasattr(self, 'cp'):
            # num_q x neg, the number of items
            item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
            item_idx = torch.floor(
                item_cnt * torch.rand_like(item_cnt.float())).int()  # num_q x neg
            neg_items = self.indices[item_idx + self.indptr[k01]] + 1
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
        fullrange = start.unsqueeze(-1) + torch.arange(
            maxlen, device=start.device).reshape(1, 1, maxlen)  # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        # @todo replace searchsorted with torch.bucketize
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(
            start.float()).unsqueeze(-1)).squeeze(-1)  # num_q x neg
        item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]]
        # plus 1 due to considering padding, since p include num_items + 1 entries
        neg_probs = self.p[item_idx + self.indptr[k01] + 1]
        return neg_items, p01 + torch.log(neg_probs)

    def compute_item_p(self, query, pos_items):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L1 || assume padding=0
        k0 = self.cd0[pos_items]  # B x L || B x L1
        k1 = self.cd1[pos_items]  # B x L || B x L1
        c0 = self.c0_[k0, :]  # B x L x D || B x L1 x D
        c1 = self.c1_[k1, :]  # B x L x D || B x L1 x D
        q0, q1 = query.chunk(2, dim=-1)  # B x L x D || B x D
        if query.dim() == pos_items.dim():
            r = (torch.bmm(c0, q0.unsqueeze(-1)) +
                 torch.bmm(c1, q1.unsqueeze(-1))).squeeze(-1)  # B x L1
        else:
            r = torch.bmm(q0, c0.transpose(1, 2)) + \
                torch.bmm(q1, c1.transpose(1, 2))
            pos_items = pos_items.unsqueeze(1)
        if not hasattr(self, 'p'):
            return r
        else:
            return r + torch.log(self.p[pos_items])


class MIDXSamplerPop(MIDXSamplerUniform):
    """
    Popularity sampling for the final items
    """

    def __init__(self, pop_count: torch.Tensor, num_clusters, scorer=None, mode=1):
        super(MIDXSamplerPop, self).__init__(
            pop_count.shape[0], num_clusters, scorer)
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_count = torch.nn.Parameter(pop_count, requires_grad=False)

    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            norm = self.pop_count
        else:
            norm = self.pop_count * \
                torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
        self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
        # self.p = torch.from_numpy(np.insert(pop_count, 0, 1.0)) 
        # # this is similar, to avoid log 0 !!! in case of zero padding
        # this is similar, to avoid log 0 !!! in case of zero padding
        self.p = torch.cat([norm.new_ones(1), norm], dim=0)
        self.cp = norm[self.indices]
        for c in range(self.K**2):
            start, end = self.indptr[c], self.indptr[c+1]
            if end > start:
                cumsum = self.cp[start:end].cumsum(0)
                self.cp[start:end] = cumsum / cumsum[-1]


class ClusterSamplerUniform(MIDXSamplerUniform):
    def __init__(self, num_items, num_clusters, scorer_fn=None):
        assert scorer_fn is None or not isinstance(scorer_fn, scorer.MLPScorer)
        super(ClusterSamplerUniform, self).__init__(
            num_items, num_clusters, scorer_fn)
        self.K = num_clusters

    def update(self, item_embs, max_iter=30):
        if isinstance(self.scorer, scorer.CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        self.c, cd, cdm, _ = kmeans(item_embs, self.K, max_iter)
        # for retreival probability, considering padding
        self.c_ = torch.cat(
            [self.c.new_zeros(1, self.c.size(1)), self.c], dim=0)
        # for retreival probability, considering padding
        self.cd = torch.cat([-cd.new_ones(1), cd], dim=0) + 1
        self.indices, self.indptr = construct_index(cd, self.K)
        self._update(item_embs, cdm)

    def _update(self, item_embs, cdm):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            self.wkk = cdm.sum(0)
        else:
            norm = torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
            self.wkk = (cdm * norm.view(-1, 1)).sum(0)
            # this is similar, to avoid log 0 !!! in case of zero padding
            self.p = torch.cat([norm.new_ones(1), norm], dim=0)
            self.cp = norm[self.indices]
            for c in range(self.K):
                start, end = self.indptr[c], self.indptr[c+1]
                if end > start:
                    cumsum = self.cp[start:end].cumsum()
                    self.cp[start:end] = cumsum / cumsum[-1]

    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad():
            if isinstance(self.scorer, scorer.CosineScorer):
                query = F.normalize(query, dim=-1)
            q = query.view(-1, query.size(-1))
            r = q @ self.c.T
            # if pos_items is not None:
            #     # Consider remove pos items
            #     N_k = self.indptr[1:] - self.indptr[:-1]
            #     pos_k_onehot = F.one_hot(self.cd[pos_items], num_classes=self.K+1)  # num_q x L x (K+1)
            #     N_k_pos = pos_k_onehot.sum(1)
            #     N_k_pos[0] = 0
            #     r += torch.log(N_k - N_k_pos)
            rs = torch.softmax(r, dim=-1)   # num_q x K
            k = torch.multinomial(
                rs, num_neg, replacement=True)    # num_q x neg
            p = torch.gather(r, -1, k)
            neg_items, neg_prob = self.sample_item(k, p, pos_items)
            pos_prop = self.compute_item_p(query, pos_items)
            return pos_prop, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def compute_item_p(self, query, pos_items):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L1 || assume padding=0
        k = self.cd[pos_items]  # B x L || B x L1
        c = self.c_[k, :]  # B x L x D || B x L1 x D
        if query.dim() == pos_items.dim():
            r = torch.bmm(c, query.unsqueeze(-1)).squeeze(-1)  # B x L1
        else:
            r = torch.bmm(query, c.transpose(1, 2))  # B x L x L1
            pos_items = pos_items.unsqueeze(1)
        if not hasattr(self, 'p'):
            return r
        else:
            return r + torch.log(self.p[pos_items])

    def sample_item(self, k01, p01, pos=None):
        if not hasattr(self, 'cp'):
            # pos_indices = self.itemid2indice[pos]   # TODO: consider remove pos case
            # num_q x neg, the number of items
            item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
            item_idx = torch.floor(
                item_cnt * torch.rand_like(item_cnt.float())).int()  # num_q x neg
            neg_items = self.indices[item_idx + self.indptr[k01]] + 1
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
        fullrange = start.unsqueeze(-1) + torch.arange(
            maxlen, device=start.device).reshape(1, 1, maxlen)  # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        # @todo replace searchsorted with torch.bucketize
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(
            start.float()).unsqueeze(-1)).squeeze(-1)  # num_q x neg
        item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]]
        # plus 1 due to considering padding, since p include num_items + 1 entries
        neg_probs = self.p[item_idx + self.indptr[k01] + 1]
        return neg_items, p01 + torch.log(neg_probs)


class ClusterSamplerPop(ClusterSamplerUniform):
    def __init__(self, pop_count: torch.Tensor, num_clusters, scorer=None, mode=1):
        super(ClusterSamplerPop, self).__init__(
            pop_count.shape[0], num_clusters, scorer)
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_count = torch.nn.Parameter(pop_count, requires_grad=False)

    def _update(self, item_embs, cdm):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            norm = self.pop_count
        else:
            norm = self.pop_count * \
                torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
        self.wkk = (cdm * norm.view(-1, 1)).sum(0)
        # this is similar, to avoid log 0 !!! in case of zero padding
        self.p = torch.cat([norm.new_ones(1), norm], dim=0)
        self.cp = norm[self.indices]
        for c in range(self.K):
            start, end = self.indptr[c], self.indptr[c+1]
            if end > start:
                cumsum = self.cp[start:end].cumsum(0)
                self.cp[start:end] = cumsum / cumsum[-1]


# TODO: avoid sampling pos items in MIDX and Cluster
# TODO aobpr sampler

# TODO: test and plot
def test():
    item_emb = torch.rand(100, 64).cuda()
    query = torch.rand(2, 10, 64).cuda()
    item_count = torch.floor(torch.rand(100) * 100).cuda()
    pos_id = torch.tensor(
        [[0, 0, 2, 5, 19, 29, 89], [21, 23, 11, 22, 44, 78, 39]]).cuda()
    # midx_uni = MIDXSamplerUniform(100,8).cuda()
    # midx_uni.update(item_emb)
    # pos_prob, neg_id, neg_prob = midx_uni(query, 10, pos_id)

    # midx_pop = MIDXSamplerPop(item_count, 100, 8).cuda()
    # midx_pop.update(item_emb)
    # pos_prob, neg_id, neg_prob = midx_pop(query, 10, pos_id)

    # cluster_uni = ClusterSamplerUniform(100, 8).cuda()
    # cluster_uni.update(item_emb)
    # pos_prob, neg_id, neg_prob = cluster_uni(query, 10, pos_id)

    # cluster_pop = ClusterSamplerPop(item_count, 100, 8).cuda()
    # cluster_pop.update(item_emb)
    # pos_prob, neg_id, neg_prob = cluster_pop(query, 10, pos_id)

    masked_uniform = MaskedUniformSampler(100)
    pos_prob, neg_id, neg_prob = masked_uniform(query, 100000, pos_id)
    print('end')

# test()


