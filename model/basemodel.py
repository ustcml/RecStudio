import torch.nn.functional as F
from utils.utils import set_color
from torch.utils.data import DataLoader
from data.dataset import MFDataset
from ann.search import Index
from ann import sampler
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import faiss
from . import scorer
from . import loss_func
import torch

class Recommender(nn.Module):

    def __init__(self, config):
        super(Recommender, self).__init__()
        self.config = config

    def init_model(self, train_data):
        pass

    def on_fit_begin(self, train_data):
        pass
    
    def on_fit_end(self, train_data):
        pass

    def on_epoch_begin(self, epoch_idx, train_data):
        pass

    def on_epoch_end(self, epoch_idx, train_data):
        pass
    
    def get_opt_parameter(self):
        return self.parameters()
    
    def set_model_mode(self, mode):
        if mode =='train':
           self.train()
        else:
           self.eval()
    
    def config_loss(self):
        pass

    def get_optimizer(self, params):
        if self.config['learner'].lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['learner'].lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['learner'].lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['learner'].lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['learner'].lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=self.config['learning_rate'])
            #if self.weight_decay > 0:
            #    self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            #self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.config['learning_rate'])
        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.config['scheduler'].lower() == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        return scheduler
    
    def agg_loss(self, losses):
        weights = None
        if isinstance(losses, tuple):
            if weights is not None:
                loss = sum(l * w for l, w in zip(losses, weights))
            else:
                loss = sum(losses)
            loss_tuple = np.array([_.item() for _ in losses])
        else:
            loss = losses
            loss_tuple = losses.item()
        return loss, loss_tuple

    def fit(self, train_data, use_fields=None, show_progress=True):
        self.fuid, self.fiid, self.frating = train_data.fuid, train_data.fiid, train_data.frating
        if use_fields is not None:
            train_data.drop_feat(use_fields)
        self.user_fields = train_data.user_feat.fields if use_fields is None \
            else train_data.user_feat.fields.intersection(use_fields)
        self.item_fields = train_data.item_feat.fields if use_fields is None \
            else train_data.item_feat.fields.intersection(use_fields)
        self.init_model(train_data)
        self.set_model_mode('train')
        self.on_fit_begin(train_data)
        optimizer = self.get_optimizer(self.get_opt_parameter())
        scheduler = self.get_scheduler(optimizer)
        loader = train_data.loader(batch_size=self.config['batch_size'], \
            num_workers=self.config['num_workers'], shuffle=True)
        for epoch_idx in range(self.config['epochs']):
            total_loss = None
            self.on_epoch_begin(epoch_idx, train_data)
            iter_data = (
                tqdm(
                    enumerate(loader),
                    total=(len(train_data) + self.config['batch_size'] -1) /self.config['batch_size'],
                    desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
                ) if show_progress else enumerate(loader))
            for _, batch_data in iter_data:
                y_h = self.forward(batch_data, isinstance(self.loss_fn, loss_func.FullScoreLoss))
                losses = self.loss_fn(batch_data[self.frating], *y_h if isinstance(y_h, tuple) else y_h)
                loss, losstuple = self.agg_loss(losses)
                total_loss = losstuple if total_loss is None else losstuple + total_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Iteration {epoch_idx}: loss={total_loss}')
            scheduler.step()
            self.on_epoch_end(epoch_idx, train_data)
        self.on_fit_end(train_data)

    def evaluate(self, test_data):
        pass
    
    def predict(self, query):
        pass

    
class TowerFreeRecommender(Recommender):
    def __init__(self):
        pass

class ItemIDTowerRecommender(Recommender):
    def __init__(self, config):
        super(ItemIDTowerRecommender, self).__init__(config)
        self.neg_count = config['negative_count']
        self.score_func = self.config_scorer()
        self.loss_fn = self.config_loss()
        self.use_index = config['ann'] is not None
        self.embed_dim = self.config['embed_dim']
    
    def config_loss(self):
        return loss_func.BPRLoss()
    
    def config_scorer(self):
        return scorer.InnerProductScorer()
    
    def init_model(self, train_data): ## need to overwrite
        if self.neg_count is not None:
            self.sampler = self.build_sampler(train_data)
        self.item_encoder = self.build_item_encoder(train_data)

    def on_epoch_begin(self, epoch_idx, train_data):
        if self.neg_count is not None:
            self.sampler.update(self.get_item_vector())
    
    def construct_query(self, batch_data): ## need to overwrite
        pass

    def forward(self, batch_data, full_score):
        pos_items = self.item_encoder(self.get_item_feat(batch_data))
        query = self.construct_query(batch_data)
        pos_score = self.score_func(query, pos_items)
        if batch_data[self.fiid].dim() > 1:
            pos_score[batch_data[self.fiid] == 0] = -float('inf')
        if self.neg_count is not None:
            pos_prob, neg_item_idx, neg_prob = self.sampler(query, self.neg_count, pos_items)
            neg_items = self.item_encoder(self.get_item_feat(neg_item_idx))
            neg_score = self.score_func(query, neg_items)
            return pos_score, pos_prob, neg_score, neg_prob
        elif full_score:
            return pos_score, self.score_func(query, self.get_item_vector())
        else:
            return pos_score
    
    def get_item_feat(self, data):
        if isinstance(data, dict): ## batch_data
            return data[self.fiid]
        else: ## neg_item_idx
            return data 

    def build_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim)
    
    def build_sampler(self, train_data):
        if 'sampler' in self.config and self.config['sampler'] is not None:
            if self.config['sampler'].lower() == 'uniform':
                output = sampler.UniformSampler(train_data.num_items-1)
            elif self.config['sampler'].lower() == 'popularity':
                output = sampler.PopularSamplerModel(train_data.item_freq[1:], self.config['item_freq_process_mode'])
            elif self.config['sampler'].lower() == 'midx_uni':
                output = sampler.SoftmaxApprSamplerUniform(train_data.num_items-1, self.config['sampler_num_centers'])
            elif self.config['sampler'].lower() == 'midx_pop':
                output = sampler.SoftmaxApprSamplerPop(train_data.item_freq[1:], self.config['sampler_num_centers'], \
                    self.config['item_freq_process_mode'])
        else:
            output = sampler.UniformSampler(train_data.num_items-1)
        return output
    
    def get_item_vector(self):
        return self.item_encoder.weight[1:]
    
    def build_ann_index(self):
        # faiss.METRIC_INNER_PRODUCT
        item_vector = self.get_item_vector().detach()
        if isinstance(self.score_func, scorer.InnerProductScorer):
            metric = faiss.METRIC_INNER_PRODUCT
        elif isinstance(self.score_func, scorer.CosineScorer):
            metric = faiss.METRIC_INNER_PRODUCT
            item_vector = F.normalize(item_vector, dim=1)
        elif isinstance(self.score_func, scorer.EuclideanScorer):
            metric = faiss.METRIC_L2
        else:
            raise ValueError(f'ANN index do not support the {type(self.score_func).__name__}')
        num_item, dim = item_vector.shape
        #if self.config['ann'] is None:
        #    self.ann_search = 'Flat' if num_item < 50000 else 'IVF%d,Flat' % int(4*np.sqrt(num_item))
        if 'HNSWx' in self.config['ann']['index']:
            self.ann_search = self.config['ann']['index'].replace('x', '32')
        elif 'IVFx' in self.config['ann']['index']:
            self.ann_search = self.config['ann']['index'].replace('x', str(int(4*np.sqrt(num_item))))
        index = faiss.index_factory(dim, self.ann_search, metric)
        if 'parameter' in self.config['ann'] and self.config['ann']['parameter'] is not None:
            for k, v in self.config['ann']['parameter'].items():
                faiss.ParameterSpace().set_index_parameter(index, k, v)
        if not index.is_trained:
            index.train(item_vector.numpy())
        index.add(item_vector.numpy())
        return index

    def on_fit_end(self, train_data):
        if self.use_index:
            self.ann_index = self.build_ann_index()
        else:
            self.item_vector = self.get_item_vector().detach()
    
    def topk(self, query, k):
        if self.use_index:
            return self.ann_index.search(F.normalize(query, dim=1).numpy(), k)
        else:
            return torch.topk(self.score_func(query, self.item_vector), k)

class ItemTowerRecommender(ItemIDTowerRecommender):   

    def construct_query(self, batch_data): ## need to overwrite
        pass

    def get_item_feat(self, data):
        if isinstance(data, dict): ## batch_data
            return dict((field, value) for field, value in data.items() if field in self.item_fields)
        else:    ## neg_item_idx
            return self.item_feat[data]

    def build_item_encoder(self, train_data): ## need to overwrite
        pass

    def init_model(self, train_data): ## need to overwrite
        super().init_model(train_data)
        self.item_feat = train_data.item_feat
    
    def get_item_vector(self):
        output = [self.item_encoder(batch) for batch in self.item_feat.loader(batch_size=1024)]
        output = torch.cat(output, dim=0)
        return output[1:]

class UserItemIDTowerRecommender(ItemIDTowerRecommender):

    def init_model(self, train_data): ## need to overwrite
        super().init_model(train_data)
        self.user_encoder = self.build_user_encoder(train_data)

    def build_user_encoder(self, train_data): # need to overwrite
        pass
    
    def get_user_feat(self, batch_data):
        if len(self.user_fields) == 1:
            return batch_data[self.fuid]
        else:
            return dict((field, value) for field, value in batch_data.items() if field in self.user_fields)

    def construct_query(self, batch_data):
        return self.user_encoder(self.get_user_feat(batch_data))
           

class UserItemTowerRecommender(ItemTowerRecommender):

    def init_model(self, train_data): ## need to overwrite
        super().init_model(train_data)
        self.user_encoder = self.build_user_encoder(train_data)

    def build_user_encoder(self, train_data): #need to overwrite
        pass
    
    def build_item_encoder(self, train_data): #need to overwrite
        pass

    def get_user_feat(self, batch_data):
        if len(self.user_fields) == 1:
            return batch_data[self.fuid]
        else:
            return dict((field, value) for field, value in batch_data.items() if field in self.user_fields)


    def construct_query(self, batch_data):
        return self.user_encoder(self.get_user_feat(batch_data))

        

    
