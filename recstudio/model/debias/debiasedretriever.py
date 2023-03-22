from .. import loss_func, scorer
from ..basemodel import BaseRetriever
from ..loss_func import FullScoreLoss
from recstudio.data import dataset
from recstudio.utils import get_model
import torch
from typing import Dict, List, Optional, Tuple, Union

class DebiasedRetriever(BaseRetriever):
    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(config, **kwargs)
        self.backbone = torch.nn.ModuleDict()

        if 'propensity' in kwargs:
            assert isinstance(kwargs['propensity'], torch.nn.Module), \
                "propensity must be torch.nn.Module"
            self.propensity = kwargs['propensity']
        else:
            self.propensity = None

        if 'discrepancy' in kwargs:
            assert isinstance(kwargs['discrepancy'], torch.nn.Module), \
                "discrepancy must be torch.nn.Module"
            self.discrepancy = kwargs['discrepancy']
        else:
            self.discrepancy = self._get_discrepancy()
            
    def _get_dataset_class():
        return dataset.TripletDataset
    
    def _init_model(self, train_data):
        self._add_backbone(train_data)
        super()._init_model(train_data)
        self.propensity = self._get_propensity(train_data) if not self.propensity else self.propensity
    
    def _get_sampler(self, train_data):
        return None

    def _get_propensity(self, train_data):
        return None

    def _get_discrepancy(self):
        if 'discrepancy' not in self.config['train'].keys():
            return None
        elif self.config['train']['discrepancy'].lower() == 'l1':
            return loss_func.L1Loss()
        elif self.config['train']['discrepancy'].lower() == 'l2':
            return loss_func.MSELoss()
        elif self.config['train']['discrepancy'].lower() == 'dcor':
            return loss_func.dCorLoss()
        elif self.config['train']['discrepancy'].lower() == 'cos':
            return scorer.CosineScorer(reduction='mean')
        else:
            raise ValueError(f"{self.config['train']['discrepancy']} is unsupportable.")

    def _add_backbone(self, train_data):
        for name in self.config['backbone'].keys():
            if name in self.backbone.keys():
                raise ValueError(f'Backbone name {name} appears more than one time.')
            model_class, model_conf = get_model(self.config['backbone'][name]['class'])
            backbone = model_class(model_conf)
            backbone._init_model(train_data)
            self.backbone[name] = backbone

    def _get_masked_batch(self, backbone_name, batch):
        return batch

    def forward(self, batch, 
                return_query=False, return_item=False, 
                return_neg_item=False, return_neg_id=False):
        query, neg_item_idx, log_pos_prob, log_neg_prob = None, None, None, None
        if self.config['train']['co_sampling']:
            if self.sampler is not None:
                if not self.neg_count:
                    raise ValueError("`negative_count` value is required when "
                                    "`sampler` is not none.")
                (log_pos_prob, neg_item_idx, log_neg_prob), query = self.sampling(batch=batch, num_neg=self.neg_count,
                                                                                excluding_hist=self.config['train'].get('excluding_hist', False),
                                                                                method=self.config['train'].get('sampling_method', 'none'), return_query=True)
        query = self.query_encoder.split(query)
        output = {}
        for name, backbone in self.backbone.items():
            masked_batch = self._get_masked_batch(name, batch)
            output[name] = backbone.forward(
                masked_batch, 
                isinstance(backbone.loss_fn, FullScoreLoss),
                return_query=True, 
                return_item=True,
                return_neg_item=True,
                return_neg_id=True,
                query=query[name], 
                neg_item_idx=neg_item_idx,
                log_pos_prob=log_pos_prob,
                log_neg_prob=log_neg_prob)
        return output

    def training_step(self, batch, nepoch=None, loader_idx=None, batch_idx=None):
        output = self.forward(batch)
        loss = {}
        for name, backbone in self.backbone.items():
            score = self._get_score(name, output, self._get_masked_batch(name, batch))
            if backbone.loss_fn is not None:
                loss[name] = backbone.loss_fn(
                    reduction=self.config['backbone'][name]['loss_reduction'],
                    **score)
        loss_value = self._get_final_loss(loss, output, batch)
        return loss_value

    def _get_score(self, name, output, batch):
        score = output[name]['score']
        score['label'] = batch[self.frating]
        return score
    
    def _get_final_loss(self, loss : Dict, output : Dict, batch : Dict):
        return sum(loss.values())
    
    
    """Below is all for evaluation."""
    def _get_query_encoder(self, train_data):
        return DebiasedQueryEncoder(self.backbone)

    def _get_item_encoder(self, train_data):
        return DebiasedItemEncoder(self.backbone)
    
    def _get_query_feat(self, data):
        query_feat = {}
        for k, v in self.backbone.items():
            query_feat[k] = v._get_query_feat(data)
        return query_feat 
      
    def _get_item_feat(self, data):
        item_feat = {}
        for k, v in self.backbone.items():
            item_feat[k] = v._get_item_feat(data)
        return item_feat
    
    def _get_item_vector(self):
        item_vector = {}
        for name, backbone in self.backbone.items():
            item_vector[name] = backbone._get_item_vector()
        item_vector = torch.hstack([v for _, v in item_vector.items()])
        return item_vector
    
class DebiasedQueryEncoder(torch.nn.Module):
    def __init__(self, backbone, 
                 concat_func=lambda d: torch.hstack([v for _, v in d.items()]),
                 split_func=lambda x, num: x.chunk(num, dim=-1)):
        """func(function): decide how to get the query vector"""
        super().__init__()
        self.backbone_names = list(backbone.keys())
        self.concat_func = concat_func
        self.split_func = split_func
        self.query_encoders = {}
        for k, v in backbone.items():
            self.query_encoders[k] = v.query_encoder
    def forward(self, input):
        """input (dict): {backbone name: corresponding query feat}"""
        query = {}
        for k, v in self.query_encoders.items():
            query[k] = v(input[k])
        query = self.concat_func(query)
        return query
    def split(self, query):
        if query is not None:
            queries = self.split_func(query, len(self.backbone_names))
            query = {k: v for k, v in zip(self.backbone_names, queries)}
        else:
            query = {k: None for k in self.backbone_names}
        return query
    
class DebiasedItemEncoder(torch.nn.Module):
    def __init__(self, backbone,
                 func=lambda d: torch.hstack([v for _, v in d.items()])):
        """choice(str): one of backbone names or `all`"""
        super().__init__()
        self.func = func
        self.item_encoders = {}
        for k, v in backbone.items():
            self.item_encoders[k] = v.item_encoder
    def forward(self, input):
        """input (dict): {backbone name: corresponding item feat}"""
        item = {}
        for k, v in self.item_encoders.items():
            item[k] = v(input[k])
        item = self.func(item)
        return item