from typing import Any, Optional
import torch.nn.functional as F
from recstudio.utils import set_color, parser_yaml, color_dict, print_logger
from recstudio.data.dataset import AEDataset, MFDataset, SeqDataset, FullSeqDataset
from recstudio.data.advance_dataset import ALSDataset
from recstudio.ann import sampler
from torch import optim
import numpy as np
import faiss, torch, os, nni, abc
import recstudio.eval as eval
from . import scorer, loss_func, init
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loops import Loop, EvaluationLoop
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.utilities.model_helpers import is_overridden
from collections import OrderedDict
# configure logging at the root level of lightning
class Recommender(LightningModule, abc.ABC):
    r"""The basic recommender model class.

    Args: 
        config(dict): configuration for the model.
    """
    def __init__(self, config):
        super(Recommender, self).__init__()
        seed_everything(42, workers=True) ## Trainer(deterministic=False) 
        self.config = config
        self.loss_fn = self.config_loss()
        self.fields = self.config.get('use_fields')
        '''@nni.variable(nni.choice(16, 32, 64), name=embed_dim)'''
        embed_dim = self.config['embed_dim']
        self.embed_dim = embed_dim
        self.save_hyperparameters(config)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        r"""Add model specific command arguments for model.

        Args:
            parent_parser(argparse.Parser): the command line parser.

        Returns:
            argparse.Parser: the parser after adding specific arguments.
        """
        parent_parser = Trainer.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("Recommender")
        parser.add_argument("--learning_rate", type=float, default=0.001, help='learning rate')
        parser.add_argument("--learner", type=str, default="adam", help='optimization algorithm')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay coefficient')
        parser.add_argument('--epochs', type=int, default=50, help='training epochs')
        parser.add_argument('--batch_size', type=int, default=2048, help='training batch size')
        parser.add_argument('--eval_batch_size', type=int, default=128, help='evaluation batch size')
        parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
        return parent_parser

    def configure_callbacks(self):
        if self.val_check:
            eval_metric = self.config['val_metrics']
            self.val_metric = next(iter(eval_metric)) if isinstance(eval_metric, list)  else eval_metric
            cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config['cutoff']]
            if len(eval.get_rank_metrics(self.val_metric)) > 0:
                self.val_metric += '@' + str(cutoffs[0])
            early_stopping = EarlyStopping(self.val_metric, verbose=True, patience=10, mode=self.config['early_stop_mode'])
            ckp_callback = ModelCheckpoint(monitor=self.val_metric, save_top_k=1, mode=self.config['early_stop_mode'], save_last=True)
            return [ckp_callback, early_stopping]
    
    @abc.abstractmethod
    def get_dataset_class(self):
        r"""Configure specific dataset type for the model.

        The optional dataset can be referred in `recstudio.dataset`.
        """
        pass

    def load_dataset(self, data_config_file):
        r"""Load dataset for the model.

        Args:
            data_config_file(str): the file path of the dataset file.

        Returns:
            list: A list contains train/valid/test data-[train, valid, test]
        """
        cls = self.get_dataset_class()
        dataset = cls(data_config_file)
        if cls in (MFDataset, ALSDataset):
            parameter = {'shuffle': self.config.get('shuffle'),
                        'split_mode': self.config.get('split_mode')}
            if isinstance(self, TowerFreeRecommender):
                parameter['fmeval'] = True
        elif cls == AEDataset:
            parameter = {'shuffle': self.config.get('shuffle')}
        elif cls in (SeqDataset, FullSeqDataset):
            parameter = {'rep' : self.config.get('test_repetitive'),
                        'train_rep': self.config.get('train_repetitive')}
        parameter['dataset_sampling_count'] = self.config.get('dataset_sampling_count')
        parameter = {k: v for k, v in parameter.items() if v is not None}
        return dataset.build(self.config['split_ratio'], **parameter)

    @abc.abstractmethod
    def init_model(self, train_data):
        """Init model with some necessary layers.

        Model structures are expected to be config here. Here is a simple example for GRU4Rec model:

        **Example**

            >>> self.hidden_size = self.config['hidden_size']
            >>> self.num_layers = self.config['layer_num']
            >>> self.dropout_rate = self.config['dropout_rate']
            >>> self.emb_dropout = torch.nn.Dropout(self.dropout_rate)
            >>> self.GRU = torch.nn.GRU(
            >>>     input_size = self.embed_dim,
            >>>     hidden_size = self.hidden_size,
            >>>     num_layers = self.num_layers,
            >>>     bias = False,
            >>>     batch_first = True,
            >>>     bidirectional = False
            >>> )
            >>> self.dense = torch.nn.Linear(self.hidden_size, self.embed_dim)

        Args:
            train_data(recstudio.data.Dataset): the train dataset. 
        """
        self.frating = train_data.frating
        if self.fields is not None:
            assert self.frating in self.fields
            train_data.drop_feat(self.fields)
        else:
            self.fields = set(f for f in train_data.field2type if 'time' not in f)
            
        self.user_fields = set(train_data.user_feat.fields).intersection(self.fields)
        self.item_fields = set(train_data.item_feat.fields).intersection(self.fields)        

    def init_parameter(self):
        r"""Init parameters in each layer if the model.
        
        `torch.nn.init.xavier_normal_` is used to initialize all the parameters.
        """
        self.apply(init.xavier_normal_initialization)


    def training_epoch_end(self, outputs):
        loss_metric = {'train_'+ k: torch.hstack([e[k] for e in outputs]).mean() for k in outputs[0]}
        self.log_dict(loss_metric, on_step=False, on_epoch=True)
        if self.val_check and self.run_mode == 'tune':
            metric = self.trainer.logged_metrics[self.val_metric]
            nni.report_intermediate_result(metric)
        if self.run_mode in ['light', 'tune'] or self.val_check:
            print_logger.info(color_dict(self.trainer.logged_metrics, self.run_mode=='tune'))
        else:
            print_logger.info('\n'+color_dict(self.trainer.logged_metrics, self.run_mode=='tune'))
    
    def validation_epoch_end(self, outputs):
        val_metric = self.config['val_metrics'] if isinstance(self.config['val_metrics'], list) else [self.config['val_metrics']]
        cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config.setdefault('cutoff', None)]
        val_metric = [f'{m}@{cutoff}' if len(eval.get_rank_metrics(m)) > 0 else m  for cutoff in cutoffs[:1] for m in val_metric]
        out = self._test_epoch_end(outputs)
        out = dict(zip(val_metric, out))
        self.log_dict(out, on_step=False, on_epoch=True)

    def test_epoch_end(self, outputs):
        test_metric = self.config['test_metrics'] if isinstance(self.config['test_metrics'], list) else [self.config['test_metrics']]
        cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config.setdefault('cutoff', None)]
        test_metric = [f'{m}@{cutoff}' if len(eval.get_rank_metrics(m)) > 0 else m for cutoff in cutoffs for m in test_metric]
        out = self._test_epoch_end(outputs)
        out = dict(zip(test_metric, out))
        self.log_dict(out, on_step=False, on_epoch=True)

    def _test_epoch_end(self, outputs):
        metric, bs = zip(*outputs)
        metric = torch.tensor(metric)
        bs = torch.tensor(bs)
        out = (metric * bs.view(-1, 1)).sum(0) / bs.sum()
        return out
        
            
    @abc.abstractmethod  
    def config_loss(self):
        r"""Config model loss.
        
        In recommendation models, several loss functions are commonly used, such as 
        SoftmaxLoss, BPRLoss, BCELoss, SampledSoftmaxLoss and so on. We provide the most common used loss function in the framework.

        To config loss, please refer to `recstudio.model.loss`
        """
        pass
    
    def get_optimizer(self, params):
        r"""Return optimizer for specific parameters.

        The optimizer can be configured in the config file with the key ``learner``. 
        Supported optimizer: ``Adam``, ``SGD``, ``AdaGrad``, ``RMSprop``, ``SparseAdam``.

        .. note::
            If no learner is assigned in the configuration file, then ``Adam`` will be user.

        Args:
            params: the parameters to be optimized.
        
        Returns:
            torch.optim.optimizer: optimizer according to the config.
        """
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
        r"""Return learning rate scheduler for the optimizer.

        Args:
            optimizer(torch.optim.Optimizer): the optimizer which need a scheduler.

        Returns:
            torch.optim.lr_scheduler: the learning rate scheduler.
        """
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

    def fit(self, train_data, val_data=None, run_mode='detail'):
        """Function to train model and make validation.

        Args:
            train_data(recstudio.data.MFDataset): train dataset.

            val_data(recstudio.data.MFDataset, optimal): validation dataset. If None, there is no validation step.. Default: `None`.

            run_mode(str, optimal): mode for training. There are 3 choices: [``tune``,``light``,``detail``]

        .. note::
            The training procedure will run in terminal in light and detail mode, while detail mode display more training details.
            
            The ``tune`` mode is used under the ``nni`` environment.  
        """
        self.run_mode = run_mode
        self.val_check = val_data is not None and self.config['val_metrics'] is not None
        if run_mode == 'tune' and "NNI_OUTPUT_DIR" in os.environ: 
            save_dir = os.environ["NNI_OUTPUT_DIR"] #for parameter tunning
        else:
            save_dir = os.getcwd()
        print_logger.info('save_dir:' + save_dir)
        refresh_rate = 0 if run_mode in ['light', 'tune'] else 1
        logger = TensorBoardLogger(save_dir=save_dir, name="tensorboard")
        iscombine = self.set_train_loaders(train_data)
        self.init_model(train_data)
        self.init_parameter()
        train_loader = train_data.train_loader(batch_size=self.config['batch_size'], \
            shuffle=True, num_workers=self.config['num_workers'], load_combine=iscombine)
        multiple_trainloader_mode = 'min_size'
        if iscombine == True and len(train_loader[0].dataset) > len(train_loader[1].dataset):
            multiple_trainloader_mode = 'max_size_cycle'
        if val_data:
            val_loader = val_data.eval_loader(batch_size=self.config['eval_batch_size'],\
                num_workers=self.config['num_workers'])
        else:
            val_loader = None
        
        trainer = Trainer(gpus=self.config['gpu'], 
                            max_epochs=self.config['epochs'], 
                            num_sanity_val_steps=0,
                            progress_bar_refresh_rate=refresh_rate,
                            multiple_trainloader_mode=multiple_trainloader_mode,
                            logger=logger,
                            accelerator="dp")
        self.config_fitloop(trainer)
        trainer.fit(self, train_loader, val_loader)

    def config_fitloop(self, trainer):
        pass

    def evaluate(self, test_data, verbose=True):
        r""" Predict for test data.

        Args: 
            test_data(recstudio.data.Dataset): The dataset of test data, which is generated by RecStudio.

            verbose(bool, optimal): whether to show the detailed information.

        Returns:
            dict: dict of metrics. The key is the name of metrics.
        """
        test_loader = test_data.eval_loader(batch_size=self.config['eval_batch_size'], \
            num_workers=self.config['num_workers'])
        output = self.trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=False)[0]
        if self.run_mode == 'tune':
            output['default'] = output[self.val_metric]
            nni.report_final_result(output)
        if verbose:
            print_logger.info(color_dict(output, self.run_mode=='tune'))
        return output
        
    def set_train_loaders(self, train_data):
        train_data.loaders = [train_data.loader]
        train_data.nepoch = None
        return False
    
class TowerFreeRecommender(Recommender):

    def get_dataset_class(self):
        return MFDataset

    def __init__(self, config):
        super().__init__(config)

    def training_step(self, batch, batch_idx):
        y_h = self.forward(batch)
        loss = self.loss_fn(y_h, (batch[self.frating]>3).float())
        return loss

    def validation_step(self, batch, batch_idx):
        eval_metric = self.config['val_metrics']
        return self._test_step(batch, eval_metric)
    
    def test_step(self, batch, batch_idx):
        eval_metric = self.config['val_metrics']
        return self._test_step(batch, eval_metric)
    
    def _test_step(self, batch, eval_metric):
        pred = self.forward(batch)
        target = (batch[self.frating]>3).float()
        eval_metric = eval.get_pred_metrics(eval_metric)
        result = [f(pred, target.int()) for n, f in eval_metric]
        return result


class ItemTowerRecommender(Recommender):
    def __init__(self, config):
        super(ItemTowerRecommender, self).__init__(config)
        self.neg_count = config['negative_count']
        self.score_func = self.config_scorer()
        self.use_index = config['ann'] is not None and \
            (not config['item_bias'] or \
                (isinstance(self.score_func, scorer.InnerProductScorer) or \
                 isinstance(self.score_func, scorer.EuclideanScorer))
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(ItemTowerRecommender, ItemTowerRecommender).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ItemTowerRecommender")
        parser.add_argument('--sampler', type=str, default='uniform', help='sampler for some loss function')
        parser.add_argument('--negative_count', type=int, default=1, help='negative count for samplers')
        return parent_parser
    
    @abc.abstractmethod
    def config_scorer(self):
        pass
    
    @abc.abstractmethod
    def init_model(self, train_data): ## need to overwrite
        super().init_model(train_data)
        self.fiid = train_data.fiid
        assert self.fiid in self.fields
        self.item_encoder = self.build_item_encoder(train_data)
        if len(self.item_fields) > 1:
            self.item_feat = train_data.item_feat
        if self.neg_count is not None:
            self.sampler = self.build_sampler(train_data)
        
    def training_step(self, batch, batch_idx):
        y_h = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
        loss = self.loss_fn(batch[self.frating], *y_h if isinstance(y_h, tuple) else y_h)
        return loss    

    def on_train_epoch_start(self):
        if self.neg_count is not None and isinstance(self.sampler, torch.nn.Module):
            self.sampler.update(self.get_item_vector().detach().clone())
        
    @abc.abstractmethod
    def construct_query(self, batch_data): ## need to overwrite
        pass

    def forward(self, batch_data, full_score):
        pos_items = self.get_item_feat(batch_data)
        query = self.construct_query(batch_data)
        pos_score = self.score_func(query, self.item_encoder(pos_items))
        if batch_data[self.fiid].dim() > 1:
            pos_score[batch_data[self.fiid] == 0] = -float('inf')
        if self.neg_count is not None:
            if isinstance(self.sampler, ItemTowerRecommender):
                sample_query = self.sampler.construct_query(batch_data)
            else:
                sample_query = query
            pos_prob, neg_item_idx, neg_prob = self.sampler(sample_query, self.neg_count, pos_items)
            neg_items = self.item_encoder(self.get_item_feat(neg_item_idx))
            neg_score = self.score_func(query, neg_items)
            return pos_score, pos_prob, neg_score, neg_prob
        elif full_score:
            return pos_score, self.score_func(query, self.get_item_vector())
        else:
            return pos_score
    
    def get_item_feat(self, data):
        if isinstance(data, dict): ## batch_data
            if len(self.item_fields) == 1:
                return data[self.fiid]
            else:
                return dict((field, value) for field, value in data.items() if field in self.item_fields)
        else: ## neg_item_idx
            if len(self.item_fields) == 1:
                return data
            else:
                return self.item_feat[data]

    def build_item_encoder(self, train_data):
        return Embedding(train_data.num_items, self.embed_dim, padding_idx=0, bias=self.config['item_bias'])
    
    def build_sampler(self, train_data):
        if 'sampler' in self.config and self.config['sampler'] is not None:
            if self.config['sampler'].lower() == 'uniform':
                output = sampler.UniformSampler(train_data.num_items-1, self.score_func)
            elif self.config['sampler'].lower() == 'popularity':
                output = sampler.PopularSamplerModel(train_data.item_freq[1:], \
                    self.score_func, self.config['item_freq_process_mode'])
            elif self.config['sampler'].lower() == 'midx_uni':
                output = sampler.MIDXSamplerUniform(train_data.num_items-1, \
                    self.config['sampler_num_centers'], self.score_func)
            elif self.config['sampler'].lower() == 'midx_pop':
                output = sampler.MIDXSamplerPop(train_data.item_freq[1:], self.config['sampler_num_centers'], \
                    self.score_func, self.config['item_freq_process_mode'])
            elif self.config['sampler'].lower() == 'cluster_uni':
                output = sampler.ClusterSamplerUniform(self.config['sampler_num_centers'], \
                    self.score_func, self.config['item_freq_process_mode'])
            elif self.config['sampler'].lower() == 'cluster_pop':
                output = sampler.ClusterSamplerPop(train_data.item_freq[1:], self.config['sampler_num_centers'], \
                    self.score_func, self.config['item_freq_process_mode'])
        else:
            output = sampler.UniformSampler(train_data.num_items-1, self.score_func)
        return output
    
    def get_item_vector(self):
        if len(self.item_fields) == 1:
            return self.item_encoder.weight[1:]
        else:
            output = [self.item_encoder(batch) for batch in self.item_feat.loader(batch_size=1024)]
            output = torch.cat(output, dim=0)
            return output[1:]
    
    def build_ann_index(self):
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

    def on_validation_start(self) -> None:
        self.prepare_testing()

    def on_test_start(self) -> None:
        self.prepare_testing()
    
    def prepare_testing(self):
        self.register_buffer('item_vector', self.get_item_vector().detach().clone())
        if self.use_index:
            self.ann_index = self.build_ann_index()
            

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        query = self.construct_query(batch)
        score, topk_items = self.topk(query, self.config['topk'], batch['user_hist'])
        return topk_items

    def test_step(self, batch, batch_idx):
        eval_metric = self.config['test_metrics']
        cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config['cutoff']]
        return self._test_step(batch, eval_metric, cutoffs)
        
    def validation_step(self, batch, batch_idx):
        eval_metric = self.config['val_metrics']
        cutoff = self.config['cutoff'][0] if isinstance(self.config['cutoff'], list) else self.config['cutoff']
        return self._test_step(batch, eval_metric, [cutoff])
    
    def _test_step(self, batch, metric, cutoffs):
        rank_m = eval.get_rank_metrics(metric)
        topk = self.config['topk']
        bs = batch[self.frating].size(0)
        assert len(rank_m)>0
        query = self.construct_query(batch)
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

class TwoTowerRecommender(ItemTowerRecommender):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(TwoTowerRecommender, TwoTowerRecommender).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("TwoTowerRecommender")
        parser.add_argument('--split_mode', type=str, default='user_entry', help='data split mode')
        return parent_parser

    def get_dataset_class(self):
        return MFDataset

    def init_model(self, train_data): ## need to overwrite
        self.fuid = train_data.fuid
        super().init_model(train_data)
        self.user_encoder = self.build_user_encoder(train_data)

    @abc.abstractmethod
    def build_user_encoder(self, train_data): # need to overwrite
        pass
    
    def get_user_feat(self, batch_data):
        if len(self.user_fields) == 1:
            return batch_data[self.fuid]
        else:
            return dict((field, value) for field, value in batch_data.items() if field in self.user_fields)

    def construct_query(self, batch_data):
        output = self.user_encoder(self.get_user_feat(batch_data))
        if self.config['item_bias']:
            if isinstance(self.score_func, scorer.InnerProductScorer):
                concate_term = torch.ones(*output.shape[:-1], 1).type_as(output)
            elif isinstance(self.score_func, scorer.EuclideanScorer):
                concate_term = torch.zeros(*output.shape[:-1], 1).type_as(output)
            else:
                raise ValueError(f"bias does not support for the score function: {type(self.score_func).__name__}")
            output = torch.cat([output, concate_term], dim=-1)
        return output


class Embedding(torch.nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, bias: bool = False, *args, **kwargs):
        if bias:
            super().__init__(num_embeddings, embedding_dim + 1, padding_idx, *args, **kwargs)
        else:
            super().__init__(num_embeddings, embedding_dim, padding_idx, *args, **kwargs)


class AllHistTrainLoop(Loop):
    
    def __init__(self, max_epochs: Optional[int] = None) -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self._results = ResultCollection(training=True)
        self.current_epoch = 0
        self.epoch_loop = TrainEpochLoop()
    
    @property
    def global_step(self) -> int:
        return self.epoch_loop.global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        self.epoch_loop.global_step = value
    
    @property
    def max_steps(self) -> int:
        return self.epoch_loop.max_steps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self.epoch_loop.max_steps = value

    @property
    def done(self) -> bool:
        stop_epochs = self.max_epochs is not None and self.current_epoch >= self.max_epochs
        return self.trainer.should_stop or stop_epochs

    def advance(self, *args, **kwargs) -> None:
        self.current_epoch += 1
        train_dataloader = self.trainer.accelerator.process_dataloader(self.trainer.train_dataloader)
        train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)
        with self.trainer.profiler.profile("run_training_epoch"):
            epoch_output = self.epoch_loop.run(train_dataloader)
            if epoch_output is None:
                return
        self.global_step -= 1
        self.trainer.logger_connector.update_train_epoch_metrics()
        self.global_step += 1
    
    def reset(self) -> None:
        pass

class TrainEpochLoop(Loop):
    def __init__(self) -> None:
        super().__init__()
        self.is_last_batch = False
        self.val_loop = EvaluationLoop()
        self.global_step = 0
        self.max_steps = None

    @property
    def done(self) -> bool:
        return self.is_last_batch

    def advance(self, train_dataloader, **kwargs) -> None:
        _, (data, self.is_last_batch) = next(train_dataloader)
        with self.trainer.profiler.profile("training_step"):
            self._epoch_output = self.trainer.accelerator.training_step(OrderedDict([("batch", data), ("batch_idx", 0)]))
            self.trainer.accelerator.post_training_step()
        

    def on_run_end(self) -> Any:
        model = self.trainer.lightning_module
        processed_epoch_output = [{'loss':self._epoch_output}]
        if is_overridden("training_epoch_end", model):
            model.training_epoch_end(processed_epoch_output)
        
        hook_name = "on_train_epoch_end"
        with self.trainer.profiler.profile(hook_name):
            # first call trainer hook
            if hasattr(self.trainer, hook_name):
                trainer_hook = getattr(self.trainer, hook_name)
                trainer_hook(processed_epoch_output)
        

        
    def on_advance_end(self) -> None:
        if self.trainer.enable_validation:
            self.trainer.validating = True
            self.val_loop.reload_evaluation_dataloaders()
            with torch.no_grad():
                self.val_loop.run()
            self.trainer.training = True
        self.global_step = self.trainer.accelerator.update_global_step(
                1, self.trainer.global_step)

    def reset(self) -> None:
        self.is_last_batch = False