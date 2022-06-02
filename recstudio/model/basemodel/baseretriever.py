from typing import Dict, List, Optional, Tuple, Union

from attr import has

import recstudio.eval as eval
import torch
import torch.nn.functional as F
from recstudio.data import dataset
from recstudio.model import loss_func
from recstudio.model import scorer
from recstudio.model.basemodel import Recommender
from recstudio.utils.utils import  print_logger
from recstudio.ann.sampler import RetriverSampler, UniformSampler, MaskedUniformSampler, Sampler, uniform_sample_masked_hist




class BaseRetriever(Recommender):
    def __init__(self, config: Dict=None, **kwargs):
        super(BaseRetriever, self).__init__(config, **kwargs)

        if 'item_encoder' in kwargs:
            assert isinstance(kwargs['item_encoder'], torch.nn.Module), \
                "item_encoder must be torch.nn.Module"
            self.item_encoder = kwargs['item_encoder']
        else:
            self.item_encoder = None

        if 'query_encoder' in kwargs:
            assert isinstance(kwargs['query_encoder'], torch.nn.Module), \
                "query_encoder must be torch.nn.Module"
            self.query_encoder = kwargs['query_encoder']
        else:
            self.query_encoder = None

        if 'scorer' in kwargs:
            assert isinstance(kwargs['scorer'], torch.nn.Module), \
                "scorer must be torch.nn.Module"
            self.score_func = kwargs['scorer']
        else:
            self.score_func = self._get_score_func()

        if 'sampler' in kwargs:
            assert isinstance(kwargs['sampler'], Sampler), \
                "sampler must be recstudio.ann.sampler.Sampler"
            self.sampler = kwargs['sampler']
        else:
            self.sampler = None


        self.use_index = self.config['ann'] is not None and \
            (not config['item_bias'] or \
                (isinstance(self.score_func, scorer.InnerProductScorer) or \
                 isinstance(self.score_func, scorer.EuclideanScorer)))

        
    def _init_model(self, train_data):
        super()._init_model(train_data)

        self.query_fields = set(train_data.user_feat.fields).intersection(train_data.use_field)
        if isinstance(train_data, dataset.AEDataset) or isinstance(train_data, dataset.SeqDataset):
            self.query_fields = self.query_fields | set(["in_"+f for f in self.item_fields])
            if isinstance(train_data, dataset.SeqDataset):
                self.query_fields = self.query_fields | set(['seqlen'])

        self.fiid = train_data.fiid
        self.fuid = train_data.fuid
        assert self.fiid in self.item_fields, 'item id is required to use.'

        self.sampler = self._get_sampler(train_data) if not self.sampler else self.sampler

        self.item_encoder = self._get_item_encoder(train_data) if not self.item_encoder else self.item_encoder
        self.query_encoder = self._get_query_encoder(train_data) if not self.query_encoder else self.query_encoder


    def _get_item_feat(self, data):
        if isinstance(data, dict): ## batch
            if len(self.item_fields) == 1:
                return data[self.fiid]
            else:
                return dict((field, value) for field, value in data.items() if field in self.item_fields)
        else: ## neg_item_idx
            if len(self.item_fields) == 1:
                return data
            else:
                return self.item_feat[data]


    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)


    def _get_query_feat(self, data):
        if isinstance(data, dict): ## batch
            if (len(self.query_fields) == 1):
                return data[list(self.query_fields)[0]]
            else:
                return dict((field, value) for field, value in data.items() if field in self.query_fields)
        else: ## neg_user as query?
            if len(self.query_fields) == 1:
                return data
            else:
                return self.user_feat[data]


    def _get_query_encoder(self, train_data):
        if self.fuid in self.query_fields:
            print_logger.warning("No specific query_encoder is configured, query_encoder "\
                "is set as Embedding for user id by default due to detect user id is in"\
                "use_fields")
            return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        else:
            print_logger.error("query_encoder missing. please configure query_encoder. if you "\
                "want to use Embedding for user id as query_encoder, please add user id "\
                "field name in use_field in your configuration.")
            raise ValueError("query_encoder missing.")


    def _get_score_func(self):
        return scorer.InnerProductScorer()


    def _get_item_vector(self):
        # if self.item_encoder is None:
        #     assert hasattr(self, 'item_vector') and self.item_vector is not None, \
        #         'model without item_encoder should have item_vector.'
        #     return self.item_vector
        if len(self.item_fields) == 1:
            return self.item_encoder.weight[1:]
        else:
            # TODO: the batch_size should be configured
            output = [self.item_encoder(self._get_item_feat(batch)) 
                for batch in self.item_feat.loader(batch_size=1024)]
            output = torch.cat(output, dim=0)
            return output[1:]


    def _update_item_vector(self): #TODO: update frequency setting
        # if not hasattr(self, 'item_vector'): # EASE based model
        #     self.item_vector = item_vector
        # else:
        item_vector = self._get_item_vector()
        self.register_buffer('item_vector', item_vector.detach().clone())

        if self.use_index:
            self.ann_index = self.build_ann_index()


    def forward(self, batch, full_score):
        pos_items = self._get_item_feat(batch)
        if self.sampler is not None:
            if self.neg_count is None:
                raise ValueError("`negative_count` value is required when "\
                    "`sampler` is not none.")

            (pos_prob, neg_item_idx, neg_prob), query = self._sample(
                batch, 
                neg = self.neg_count,
                excluding_hist = self.config['excluding_hist']
            )
            pos_score = self.score_func(query, self.item_encoder(pos_items))
            if batch[self.fiid].dim() > 1:
                pos_score[batch[self.fiid] == 0] = -float('inf') # padding

            neg_items = self._get_item_feat(neg_item_idx)
            neg_score = self.score_func(query, self.item_encoder(neg_items))
            return pos_score, pos_prob, neg_score, neg_prob
        else:
            query = self.query_encoder(self._get_query_feat(batch))
            pos_score = self.score_func(query, self.item_encoder(pos_items))
            if batch[self.fiid].dim() > 1:
                pos_score[batch[self.fiid] == 0] = -float('inf') # padding
            if full_score:
                #TODO: item_vectors here to calculate should be real-time?
                item_vectors = self._get_item_vector()
                all_item_scores = self.score_func(query, item_vectors)
                return pos_score, all_item_scores
            else:
                return (pos_score, )


    def _sample(
        self, 
        batch, 
        neg = 1, 
        excluding_hist: bool = False, 
        ):
        if hasattr(self, 'query_encoder'):
            # Retriever
            query = self.query_encoder(self._get_query_feat(batch))
        else:
            # Ranker
            query = torch.zero(
                (batch[self.frating].size(0), 1), 
                device=batch[self.frating].device
            )
            assert isinstance(self.sampler, RetriverSampler) \
                or isinstance(self.sampler, UniformSampler) \
                or isinstance(self.sampler, MaskedUniformSampler),\
                "sampler for ranker must be retriever or uniform sampler."

        if excluding_hist:
            if not "user_hist" in batch:
                print_logger.warning("`user_hist` are not in batch data, so the \
                    target item will be used as user_hist.")
                user_hist = batch['user_hist']
            else:
                #TODO: user hist v.s. pos item
                user_hist = batch[self.fiid]
        else:
            user_hist = None

        if isinstance(self.sampler, Sampler):
            #TODO: mask user_hist for sampler.Sampler
            kwargs = {
                'num_neg': neg, 
                'pos_items': batch[self.fiid], 
            }
            if isinstance(self.sampler, RetriverSampler):
                kwargs.update({
                    'batch': batch, 
                    'user_hist': user_hist,
                })

            else:
                kwargs['query'] = query

            if hasattr(self.sampler, "_update"):
                self.sampler._update()  #TODO: add frequency
            pos_prob, neg_id, neg_prob = self.sampler(**kwargs)

        else:
            raise TypeError("`sampler` only support Sampler type.")        
        return (pos_prob, neg_id, neg_prob), query



    def sampling(self, batch, num_neg, pos_items, user_hist=None, method='ips', excluding_hist=False, t=1):
        query = self.query_encoder(self._get_query_feat(batch))
        if method == 'brute': # brute force sampling
            # TODO: maybe item vector should be updated
            item_vectors = self.item_vector
            all_score = self.score_func(query, item_vectors) / t
            all_prob = torch.softmax(all_score, dim=-1)
            all_prob = F.pad(all_prob, pad=(1,0))   # padding
            pos_prob = torch.gather(all_prob, dim=-1, index=pos_items)
            if excluding_hist:
                row_index = torch.arange(
                    user_hist.size(0), 
                    device=pos_items.device
                )
                row_index = row_index.view(-1, 1).repeat(1, pos_items.size(1))
                sampling_prob = all_prob
                sampling_prob[row_index, pos_items] = 0.0
            else:
                sampling_prob = all_prob
            sampled_idx = torch.multinomial(sampling_prob, num_neg * pos_items.size(-1), replacement=True)
            neg_id = sampled_idx
            neg_prob = torch.gather(sampling_prob, dim=-1, index=sampled_idx)

        elif method == 'ips' or method == 'dns':
            if isinstance(num_neg, int):
                num_neg = [num_neg, num_neg]
            elif isinstance(num_neg, Union[List, Tuple]):
                assert len(num_neg) == 2, "length of negative_count must be 2 \
                    when it's list type for retriever_dns sampler."
                assert num_neg[0] >= num_neg[1], "the first element of \
                    negative_count must be larger than the second element."
            else:
                raise TypeError("num_neg only support int and List/Tuple type.")
            
            log_pos_prob, neg_id_pool, log_neg_prob_pool = self._sample(batch, num_neg[0], excluding_hist)

            neg_item_vector = self.item_encoder(self._get_item_feat(neg_id_pool))
            scores_on_pool_items = self.score_func(query, neg_item_vector)
            
            if method == 'dns':
                topk_id = torch.topk(scores_on_pool_items, num_neg[1])
                neg_id = torch.gather(neg_id_pool, dim=-1, index=topk_id)
                neg_prob = torch.ones_like(neg_id)
                pos_prob = torch.ones_like(pos_items)
            else:
                pos_item_vector = self.item_encoder(self._get_item_feat(batch))
                pos_score = self.score_func(query, pos_item_vector)
                pos_prob = pos_score - log_pos_prob
                probs_on_pool_items = torch.softmax(
                    scores_on_pool_items+torch.finfo(torch.float32).eps, dim=-1)
                resampled_id = torch.multinomial(probs_on_pool_items, num_neg[1], replacement=True)
                neg_id = torch.gather(neg_id_pool, dim=-1, index=resampled_id)
                neg_prob = torch.gather(scores_on_pool_items, dim=-1, index=resampled_id) - \
                    torch.gather(log_neg_prob_pool, dim=-1, index=resampled_id)
        
        return pos_prob, neg_id, neg_prob



    def build_index(self):
        raise NotImplementedError("build_index  for ranker not implemented.")


    def topk(self, query, k, user_h):
        # TODO: complete topk with retriever
        # if self.config['retriever'] is None:
        #     # topk on all items
        #     pass
        # elif self.config['retriever']=='uniform':
        #     # unform(num_items)
        #     pass
        # elif self.config['retriever']=='retriever':
        #     # retriever_query = self.retriever.query_encoder(batch)
        #     score, topk_items = self.retriever.topk(batch, k0, user_h)
        

        more = user_h.size(1) if user_h is not None else 0
        if self.use_index:
            if isinstance(self.score_func, scorer.CosineScorer):
                score, topk_items = self.ann_index.search(
                    torch.nn.functional.normalize(query, dim=1).numpy(), k + more)
            else:
                score, topk_items = self.ann_index.search(query.numpy(), k + more)
        else:
            score, topk_items = torch.topk(self.score_func(query, self.item_vector), k + more)
        topk_items += 1
        if user_h is not None:
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score1, idx = score.topk(k)
            return score1, torch.gather(topk_items, 1, idx)
        else:
            return score, topk_items


    def _get_score_func(self):
        return scorer.InnerProductScorer()


    def training_step(self, batch):
        y_h = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
        loss_value = self.loss_fn(batch[self.frating], *y_h)
        del y_h
        return loss_value
    

    def validation_step(self, batch):
        eval_metric = self.config['val_metrics']
        cutoff = self.config['cutoff'][0] if isinstance(self.config['cutoff'], list) else self.config['cutoff']
        return self._test_step(batch, eval_metric, [cutoff])


    def test_step(self, batch):
        eval_metric = self.config['test_metrics']
        cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config['cutoff']]
        return self._test_step(batch, eval_metric, cutoffs)


    def _test_step(self, batch, metric, cutoffs):
        rank_m = eval.get_rank_metrics(metric)
        topk = self.config['topk']
        bs = batch[self.frating].size(0)
        assert len(rank_m)>0
        query = self.query_encoder(self._get_query_feat(batch))
        score, topk_items = self.topk(query, topk, batch['user_hist'])
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
    