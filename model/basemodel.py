from typing import Any, Sized, Iterator, Optional
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
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import eval
class Recommender(LightningModule):

    def __init__(self, config):
        super(Recommender, self).__init__()
        seed_everything(42, workers=True) ## Trainer(deterministic=False) 
        self.config = config
        self.loss_fn = self.config_loss()
        self.trainer = Trainer(gpus=config['gpu'], 
                                max_epochs=config['epochs'],
                                num_sanity_val_steps=0)

    def configure_callbacks(self):
        if self.val_check:
            early_stopping = EarlyStopping('val_loss', verbose=True)
            ckp_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', save_last=True,\
                filename='{epoch:02d}-{val_loss:.2f}')
            return [ckp_callback, early_stopping]
        

    def init_model(self, train_data):
        pass

    def training_step(self, batch, batch_idx):
        y_h = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
        loss = self.loss_fn(batch[self.frating], *y_h if isinstance(y_h, tuple) else y_h)
        return loss

    def training_epoch_end(self, outputs):
        loss = torch.hstack([e['loss'] for e in outputs]).sum()
        self.log('train_loss', loss, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)
    
    def validation_epoch_end(self, outputs):
        pass
        #metric = torch.hstack(outputs).sum()
        #self.log('val_loss', loss, prog_bar=True)

        
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
            optimizer = optim.Adam(params, lr=self.config['learning_rate'])
        return optimizer

    def get_scheduler(self, optimizer):
        if self.config['scheduler'].lower() == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        elif self.config['scheduler'].lower() == 'onplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        else:
            scheduler = None
        return scheduler
    
    def configure_optimizers(self):
        params = self.parameters()
        optimizer = self.get_optimizer(params)
        scheduler = self.get_scheduler(optimizer)
        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                    'strict': False
                }
            }
        else:
            return optimizer

    def fit(self, train_data, use_fields=None, val_data=None, show_progress=True):
        self.val_check = val_data is not None
        self.fuid, self.fiid, self.frating = train_data.fuid, train_data.fiid, train_data.frating
        if use_fields is not None:
            train_data.drop_feat(use_fields)
        self.user_fields = train_data.user_feat.fields if use_fields is None \
            else train_data.user_feat.fields.intersection(use_fields)
        self.item_fields = train_data.item_feat.fields if use_fields is None \
            else train_data.item_feat.fields.intersection(use_fields)
        self.init_model(train_data)
        train_loader = train_data.loader(batch_size=self.config['batch_size'], shuffle=True)
        if val_data:
            val_loader = val_data.eval_loader(batch_size=self.config['eval_batch_size'])
        else:
            val_loader = None
        self.trainer.fit(self, train_loader, val_loader)

    def evaluate(self, test_data):
        test_loader = test_data.eval_loader(batch_size=self.config['eval_batch_size'])
        self.user_hist = test_data.user_hist
        self.trainer.test(self, test_loader)
        
    

    
class TowerFreeRecommender(Recommender):
    def __init__(self):
        pass

    def evaluate(self, test_data, train_data):
        pass

class ItemIDTowerRecommender(Recommender):
    def __init__(self, config):
        super(ItemIDTowerRecommender, self).__init__(config)
        self.neg_count = config['negative_count']
        self.score_func = self.config_scorer()
        self.use_index = False #config['ann'] is not None
        self.embed_dim = self.config['embed_dim']
    
    def config_loss(self):
        return loss_func.BPRLoss()
    
    def config_scorer(self):
        return scorer.InnerProductScorer()
    
    def init_model(self, train_data): ## need to overwrite
        self.item_encoder = self.build_item_encoder(train_data)
        if self.neg_count is not None:
            self.sampler = self.build_sampler(train_data)
            self.sampler.update(self.get_item_vector().detach().clone())
        

    def on_train_epoch_end(self, unused: Optional[Any] = None):
        if self.neg_count is not None:
            self.sampler.update(self.get_item_vector().detach().clone())
    
    def on_validation_epoch_start(self):
        self.prepare_testing()

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
                output = sampler.UniformSampler(train_data.num_items-1, self.score_func)
            elif self.config['sampler'].lower() == 'popularity':
                output = sampler.PopularSamplerModel(train_data.item_freq[1:], \
                    self.score_func, self.config['item_freq_process_mode'])
            elif self.config['sampler'].lower() == 'midx_uni':
                output = sampler.SoftmaxApprSamplerUniform(train_data.num_items-1, \
                    self.config['sampler_num_centers'], self.score_func)
            elif self.config['sampler'].lower() == 'midx_pop':
                output = sampler.SoftmaxApprSamplerPop(train_data.item_freq[1:], self.config['sampler_num_centers'], \
                    self.score_func, self.config['item_freq_process_mode'])
        else:
            output = sampler.UniformSampler(train_data.num_items-1, self.score_func)
        return output
    
    def get_item_vector(self):
        return self.item_encoder.weight[1:]
    
    def build_ann_index(self):
        # faiss.METRIC_INNER_PRODUCT
        #item_vector = self.get_item_vector().detach().clone()
        item_vector = self.item_vector
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

    def on_fit_end(self):
        self.prepare_testing()
    
    def prepare_testing(self):
        self.item_vector = self.get_item_vector().detach().clone()
        if self.use_index:
            self.ann_index = self.build_ann_index()
            

    def test_step(self, batch, batch_idx):
        eval_metric = self.config['eval_metrics']
        pred_m, rank_m = eval.split_metrics(eval_metric)
        if len(rank_m)>0:
            query = self.construct_query(batch)
            user_h = batch['user_hist']
            _, topk_items = self.topk(query, self.config['topk'] + user_h.size(1))
            # todo preprocess topk_items and user_h
            for n, m in rank_m:
                self.log(n, m(topk_items, user_h))
        else:
            y_ = self.forward(batch, False)
            y = batch[self.fiid]
            for n, m in pred_m:
                self.log(n, m(y, y_))
        
        #train_items = user_hist[batch[self.fuid]]

    # def evaluate(self, test_data, train_data=None):
    #     loader = test_data.loader(batch_size=self.config['eval_batch_size'], shuffle=False)
    #     if train_data:
    #         user_hist = train_data.get_hist(isUser=True)
    #         user_count = np.array([len(e) for e in user_hist])
    #     for batch in loader:
            
    
    def topk(self, query, k):
        if self.use_index:
            if isinstance(self.score_func, scorer.CosineScorer):
                return self.ann_index.search(F.normalize(query, dim=1).numpy(), k)
            else:
                return self.ann_index.search(query.numpy(), k)
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

        

    
