import abc
from typing import Dict, List, Optional, Tuple, Union
import recstudio.eval as eval
import torch
from recstudio.model.basemodel import Recommender
from recstudio.ann.sampler import UniformSampler


# 1. 负样本应该由数据集提供？
# 2. 如果由数据集提供，应该是point-wise形式吗?考虑到eval时候需要用到

class BaseRanker(Recommender):

    def _init_model(self, train_data):
        # set use_field in dataset as all field
        train_data.use_field = train_data.field - set([train_data.ftime])
        super()._init_model(train_data)
        self.scorer = self._get_scorer(train_data)
        if self.retriever is None:
            self.retriever = self._get_retriever(train_data)
        
        self.rating_threshold = train_data.config['ranker_rating_threshold']


    # def _get_inter_feat(self, batch):

    def _get_retriever(self, train_data):
        return None


    def _get_scorer(self, train_data):
        return None


    def forward(self, batch):
        # calculate scores
        #TODO: neg in dataset and from retriever, use loss in PRIS
        if self.retriever is None:
            score = self.scorer(batch)
            return (score, )
        else:
            # only positive samples in batch
            assert self.neg_count is not None, 'expecting neg_count is not None.'
            pos_prob, neg_item_idx, neg_prob = self.retriever.sampling(batch, \
                self.neg_count, batch[self.fiid], method=self.config['retrieve_method'])
            pos_score = self.scorer(batch)

            neg_items = self._get_item_feat(neg_item_idx)
            neg_batch = {}
            for k in batch.keys():
                if k not in neg_items:
                    neg_batch[k] = batch[k].view(1, *batch[k].shape).expand(\
                        self.neg_count, *tuple([-1] for i in range(len(batch[k].shape))))
                    neg_batch[k] = neg_batch[k].view(-1, *batch[k].shape)
                else:
                    neg_batch[k] = neg_items[k]

            if isinstance(neg_items, Dict):
                batch.update(neg_items)
            else:
                batch[self.fiid] = neg_items

            neg_score = self.scorer(batch)
            return pos_score, pos_prob, neg_score, neg_prob            


    def build_index(self):
        raise NotImplementedError("build_index for ranker not implemented now.")


    def training_step(self, batch):
        y_h = self.forward(batch)
        target = (batch[self.frating]>self.rating_threshold).float()
        loss = self.loss_fn(*y_h, target)
        return loss


    def validation_step(self, batch):
        eval_metric = self.config['val_metrics']
        return self._test_step(batch, eval_metric)
    

    def test_step(self, batch):
        eval_metric = self.config['val_metrics']
        return self._test_step(batch, eval_metric)
    

    def _test_step(self, batch, eval_metric):
        if self.retriever is None:
            pred = self.forward(batch)
            target = (batch[self.frating]>self.rating_threshold).float() # TODO: set rating as 0/1 in dataset
            eval_metric = eval.get_pred_metrics(eval_metric)
            result = [f(*pred, target.int()) for n, f in eval_metric]
            return result
        else:
            # TODO:  
            pass
    