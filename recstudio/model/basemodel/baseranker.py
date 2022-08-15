from typing import Dict, List, Optional, Tuple, Union
from recstudio.ann import sampler
import recstudio.eval as eval
import torch
from recstudio.model.basemodel import Recommender
from recstudio.ann.sampler import UniformSampler


class BaseRanker(Recommender):

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.fiid = train_data.fiid
        self.fuid = train_data.fuid
        self.scorer = self._get_scorer(train_data)
        if self.retriever is None:
            self.retriever = self._get_retriever(train_data)
        self.rating_threshold = train_data.config.get('ranker_rating_threshold', 0)
        if self.retriever is None:
            self.logger.warning('No retriever is used, topk metrics is not supported.')

    def _set_data_field(self, data):
        data.use_field = data.field - set([data.ftime])

    def _get_retriever(self, train_data):
        return None

    def _get_scorer(self, train_data):
        return None

    def _generate_neg_batch(self, batch, neg_id):
        num_neg = neg_id.size(-1)
        neg_id = neg_id.view(-1)
        neg_items = self._get_item_feat(neg_id)
        neg_batch = {}
        for k in batch.keys():
            if isinstance(neg_items, torch.Tensor) or (k not in neg_items):
                neg_batch[k] = batch[k].unsqueeze(1).expand(-1, num_neg,
                                                            *tuple([-1 for i in range(len(batch[k].shape)-1)]))
                neg_batch[k] = neg_batch[k].reshape(-1, *(batch[k].shape[1:]))
            else:
                neg_batch[k] = neg_items[k]

        if isinstance(neg_items, Dict):
            neg_batch.update(neg_items)
        else:
            neg_batch[self.fiid] = neg_id
        return neg_batch

    def forward(self, batch):
        # calculate scores
        # TODO: neg in dataset and from retriever, use loss in PRIS
        if self.retriever is None:
            target = (batch[self.frating] > self.rating_threshold).float()
            score = self.scorer(batch)
            return {'input': score, 'target': target}
        else:
            # only positive samples in batch
            assert self.neg_count is not None, 'expecting neg_count is not None.'
            if isinstance(self.retriever, UniformSampler):
                bs = batch[self.fiid].size(0)
                pos_prob, neg_item_idx, neg_prob = self.retriever(bs, self.neg_count, batch[self.fiid])
            else:
                pos_prob, neg_item_idx, neg_prob = self.retriever.sampling(
                    batch, self.neg_count, batch[self.fiid],
                    method=self.config['retrieve_method'])
            pos_score = self.scorer(batch).view(-1, 1)

            neg_batch = self._generate_neg_batch(batch, neg_item_idx)
            neg_score = self.scorer(neg_batch).view(-1, self.neg_count)
            return {'pos_score': pos_score, 'log_pos_prob': pos_prob, 'neg_score': neg_score, 'log_neg_prob': neg_prob}

    def score(self, batch, neg_id=None):
        # designed for cascade algorithm like RankFlow and CoRR
        if neg_id is not None:
            neg_batch = self.scorer._generate_neg_batch(batch, neg_id)
            num_neg = neg_id.size(-1)
            return self.scorer(neg_batch).view(-1, num_neg)
        else:
            return self.scorer(batch)

    def build_index(self):
        raise NotImplementedError("build_index for ranker not implemented now.")

    def training_step(self, batch):
        y_h = self.forward(batch)
        loss = self.loss_fn(**y_h)
        return loss

    def validation_step(self, batch):
        eval_metric = self.config['val_metrics']
        if self.config['cutoff'] is not None:
            cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], List) \
                else [self.config['cutoff']]
        else:
            cutoffs = None
        return self._test_step(batch, eval_metric, cutoffs)

    def test_step(self, batch):
        eval_metric = self.config['test_metrics']
        if self.config['cutoff'] is not None:
            cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], List) \
                else [self.config['cutoff']]
        else:
            cutoffs = None
        return self._test_step(batch, eval_metric, cutoffs)

    def topk(self, batch, topk, user_hist=None, return_retriever_score=False):
        if self.retriever is None:
            raise NotImplementedError("`topk` function not supported for ranker without retriever.")
        else:
            score_re, topk_items_re = self.retriever.topk(batch, topk, user_hist)
            topk_batch = self._generate_neg_batch(batch, topk_items_re)
            score = self.scorer(topk_batch).view(-1, topk)
            _, sorted_idx = score.sort(dim=-1, descending=True)
            topk_items = torch.gather(topk_items_re, -1, sorted_idx)
            score = torch.gather(score_re, -1, sorted_idx)
            if return_retriever_score:
                return score, topk_items, score_re, topk_items_re
            else:
                return score, topk_items

    def _test_step(self, batch, metric, cutoffs=None):
        rank_m = eval.get_rank_metrics(metric)
        pred_m = eval.get_pred_metrics(metric)
        bs = batch[self.frating].size(0)
        if len(rank_m) > 0:
            assert cutoffs is not None, 'expected cutoffs for topk ranking metrics.'

        # TODO: discuss in which cases pred_metrics should be calculated. According to whether there are neg labels in dataset?
        # When there are neg labels in dataset, should rank_metrics be considered?
        if self.retriever is None:
            result = self.forward(batch)
            result = [f(result['input'], result['target'].int()) for n, f in pred_m], bs
            return result
        else:
            topk = self.config['topk']
            score, topk_items = self.topk(batch, topk, batch['user_hist'])
            if batch[self.fiid].dim() > 1:
                target, _ = batch[self.fiid].sort()
                idx_ = torch.searchsorted(target, topk_items)
                idx_[idx_ == target.size(1)] = target.size(1) - 1
                label = torch.gather(target, 1, idx_) == topk_items
                pos_rating = batch[self.frating]
            else:
                label = batch[self.fiid].view(-1, 1) == topk_items
                pos_rating = batch[self.frating].view(-1, 1)
            return [func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m], bs
