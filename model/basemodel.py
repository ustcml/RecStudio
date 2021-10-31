from typing import Any, Sized, Iterator, Optional
import torch.nn.functional as F
from utils.utils import set_color, parser_yaml, color_dict
from torch.utils.data import DataLoader
from data.dataset import AEDataset, MFDataset, SeqDataset
from ann.search import Index
from ann import sampler
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import faiss, torch, eval, os, logging
from . import scorer, loss_func, init
from pytorch_lightning import Trainer, LightningModule, seed_everything, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint 
from pytorch_lightning.loggers import TensorBoardLogger
# configure logging at the root level of lightning
print_logger = logging.getLogger("pytorch_lightning")
print_logger.setLevel(logging.INFO)
class Recommender(LightningModule):

    def __init__(self, config):
        super(Recommender, self).__init__()
        seed_everything(42, workers=True) ## Trainer(deterministic=False) 
        self.config = config
        self.loss_fn = self.config_loss()
        self.fields = self.config.get('use_fields')
        self.save_hyperparameters(config)

    def configure_callbacks(self):
        if self.val_check:
            self.val_metric = next(iter(self.config['val_metrics'])) + '@' + str(self.config['cutoff'])
            early_stopping = EarlyStopping(self.val_metric, verbose=True, patience=5, mode=self.config['early_stop_mode'])
            ckp_callback = ModelCheckpoint(monitor=self.val_metric, save_top_k=1, mode=self.config['early_stop_mode'], save_last=True)
            return [ckp_callback, early_stopping]

    def get_dataset_class(self):
        pass

    def load_dataset(self, data_config_file):
        cls = self.get_dataset_class()
        dataset = cls(data_config_file)
        if cls == MFDataset:
            parameter = {'shuffle': self.config.get('shuffle'),
                        'split_mode': self.config.get('split_mode')}
        elif cls == AEDataset:
            parameter = {'shuffle': self.config.get('shuffle')}
        elif cls == SeqDataset:
            parameter = {'rep' : self.config.get('test_repetitive'),
                        'train_rep': self.config.get('train_repetitive')}
        parameter = {k: v for k, v in parameter.items() if v is not None}
        return dataset.build(self.config['split_ratio'], **parameter)

    def init_model(self, train_data):
        pass

    def init_parameter(self):
        self.apply(init.xavier_normal_initialization)

    def training_step(self, batch, batch_idx):
        y_h = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
        loss = self.loss_fn(batch[self.frating], *y_h if isinstance(y_h, tuple) else y_h)
        return loss

    def training_epoch_end(self, outputs):
        loss = torch.hstack([e['loss'] for e in outputs]).sum()
        self.log('train_loss', loss)
        if self.val_check:
            metric = self.trainer.logged_metrics[self.val_metric]
            '''@nni.report_intermediate_result(metric)'''
        if self.run_mode in ['light', 'tune'] or self.val_check:
            print_logger.info(color_dict(self.trainer.logged_metrics, self.run_mode=='tune'))
        else:
            print_logger.info('\n'+color_dict(self.trainer.logged_metrics, self.run_mode=='tune'))
            
        
    
    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)
           
    def test_epoch_end(self, outputs):
        rec = outputs[0]
        for i in range(len(rec)):
            length = 0
            total = 0
            for _ in outputs:
                n, v, bs = _[i]
                total += v * bs
                length += bs
            self.log(n, total/length)
        
    def config_loss(self):
        pass
    
    def get_optimizer(self, params):
        '''@nni.variable(nni.choice(0.1, 0.05, 0.01, 0.005, 0.001), name=learning_rate)'''
        learning_rate = self.config['learning_rate']
        '''@nni.variable(nni.choice(0.1, 0.01, 0.001, 0), name=decay)'''
        decay = self.config['weight_decay']
        if self.config['learner'].lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
        elif self.config['learner'].lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=decay)
        elif self.config['learner'].lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=decay)
        elif self.config['learner'].lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=decay)
        elif self.config['learner'].lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            #if self.weight_decay > 0:
            #    self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def get_scheduler(self, optimizer):
        if self.config['scheduler'] is not None:
            if self.config['scheduler'].lower() == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            elif self.config['scheduler'].lower() == 'onplateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            else:
                scheduler = None
        else:
            scheduler = None
        return scheduler
    
    def configure_optimizers(self):
        params = self.parameters()
        optimizer = self.get_optimizer(params)
        scheduler = self.get_scheduler(optimizer)
        m = self.val_metric if self.val_check else 'train_loss'
        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': m,
                    'interval': 'epoch',
                    'frequency': 1,
                    'strict': False
                }
            }
        else:
            return optimizer

    def fit(self, train_data, val_data=None, run_mode='detail'): #mode='tune'|'light'|'detail'
        self.run_mode = run_mode
        self.val_check = val_data is not None and self.config['val_metrics'] is not None
        if run_mode == 'tune' and "NNI_OUTPUT_DIR" in os.environ: 
            save_dir = os.environ["NNI_OUTPUT_DIR"] #for parameter tunning
        else:
            save_dir = os.getcwd()
        print_logger.info('save_dir:' + save_dir)
        refresh_rate = 0 if run_mode in ['light', 'tune'] else 1
        logger = TensorBoardLogger(save_dir=save_dir, name="tensorboard")
        trainer = Trainer(gpus=self.config['gpu'], 
                            max_epochs=self.config['epochs'], 
                            num_sanity_val_steps=0,
                            progress_bar_refresh_rate=refresh_rate,
                            logger=logger)
        self.frating = train_data.frating
        if self.fields is not None:
            assert self.frating in self.fields
            train_data.drop_feat(self.fields)
        self.user_fields = train_data.user_feat.fields if self.fields is None \
            else train_data.user_feat.fields.intersection(self.fields)
        self.item_fields = train_data.item_feat.fields if self.fields is None \
            else train_data.item_feat.fields.intersection(self.fields)
        self.init_model(train_data)
        self.init_parameter()
        train_loader = train_data.loader(batch_size=self.config['batch_size'], shuffle=True)
        if val_data:
            val_loader = val_data.eval_loader(batch_size=self.config['eval_batch_size'])
        else:
            val_loader = None
        trainer.fit(self, train_loader, val_loader)

    def evaluate(self, test_data):
        test_loader = test_data.eval_loader(batch_size=self.config['eval_batch_size'])
        output = self.trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=False)[0]
        metric = output[self.val_metric]
        '''@nni.report_final_result(metric)'''
        print_logger.info(color_dict(output, self.run_mode=='tune'))
        
    

    
class TowerFreeRecommender(Recommender):
    def __init__(self):
        pass

    def evaluate(self, test_data, train_data):
        pass

    def validation_step(self, batch, batch_idx):
        eval_metric = self.config['eval_metrics']
        pred_m, _ = eval.split_metrics(eval_metric)
        bs = batch[self.frating].size(0)
        y_ = self.forward(batch, False)
        y = batch[self.frating]
        return [[n, m(y, y_), bs] for n, m in pred_m]

class ItemIDTowerRecommender(Recommender):
    def __init__(self, config):
        super(ItemIDTowerRecommender, self).__init__(config)
        self.neg_count = config['negative_count']
        self.score_func = self.config_scorer()
        self.use_index = False #config['ann'] is not None
        '''@nni.variable(nni.choice(16, 32, 64), name=embed_dim)'''
        embed_dim = self.config['embed_dim']
        self.embed_dim = embed_dim
    
    def config_loss(self):
        return loss_func.BPRLoss()
    
    def config_scorer(self):
        return scorer.InnerProductScorer()
    
    def init_model(self, train_data): ## need to overwrite
        self.fiid = train_data.fiid
        assert self.fiid in self.fields
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
        eval_metric = self.config['test_metrics']
        cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config['cutoff']]
        return self._test_step(batch, eval_metric, cutoffs)
        
    def validation_step(self, batch, batch_idx):
        eval_metric = self.config['val_metrics']
        cutoff = self.config['cutoff'][0] if isinstance(self.config['cutoff'], list) else self.config['cutoff']
        return self._test_step(batch, eval_metric, [cutoff])
    
    def _test_step(self, batch, metric, cutoffs):
        _, rank_m = eval.split_metrics(metric)
        topk = self.config['topk']
        bs = batch[self.frating].size(0)
        assert len(rank_m)>0
        query = self.construct_query(batch)
        score, topk_items = self.topk(query, topk, batch['user_hist'])
        target, _ = batch[self.fiid].sort()
        idx_ = torch.searchsorted(target, topk_items)
        idx_[idx_ == target.size(1)] = target.size(1) - 1
        label = torch.gather(target, 1, idx_) == topk_items
        return [[f'{name}@{cutoff}', func(label, batch[self.frating], cutoff), bs]\
            for cutoff in cutoffs for name, func in rank_m]
    
    
    def topk(self, query, k, user_h):
        more = user_h.size(1) if user_h is not None else 0
        if self.use_index:
            if isinstance(self.score_func, scorer.CosineScorer):
                score, topk_items = self.ann_index.search(F.normalize(query, dim=1).numpy(), k + more)
            else:
                score, topk_items = self.ann_index.search(query.numpy(), k + more)
        else:
            score, topk_items = torch.topk(self.score_func(query, self.item_vector), k + more)
        if user_h is not None:
            topk_items += 1
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score1, idx = score.topk(k)
            return score1, torch.gather(topk_items, 1, idx)
        else:
            return score, topk_items

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

    def get_dataset_class(self):
        return MFDataset

    def init_model(self, train_data): ## need to overwrite
        self.fuid = train_data.fuid
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

    def get_dataset_class(self):
        return MFDataset

    def init_model(self, train_data): ## need to overwrite
        self.fuid = train_data.fuid
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

        

    
