import torch, os
from recstudio.model import basemodel, scorer, loss_func
from recstudio.data import dataset
from recstudio.ann import sampler as samplers
from recstudio.utils.utils import color_dict, print_logger, seed_everything



class SequentialRetriever(basemodel.BaseRetriever):
    def __init__(
        self, 
        config, 
        dataset_class = dataset.MFDataset,
        ):
        super().__init__(config)
        # self.config = config
        self.dataset_class = dataset_class
        

    def configure(self,
        train_data,
        loss_func,
        item_encoder = None, 
        query_encoder = None, 
        score_func = scorer.InnerProductScorer(),
        sampler: str = None
    ):

        if item_encoder is not None:
            self.item_encoder = item_encoder
        else:
            self.item_encoder = super()._get_item_encoder(train_data)

        if query_encoder is not None:
            self.query_encoder = query_encoder
        else:
            self.query_encoder = super()._get_query_encoder(train_data)
        
        self.score_func = score_func
        self.loss_fn = loss_func
        self.sampler = sampler

        # if sampler is not None:
        #     self.config['sampler'] = sampler
        
        self.config_sampler(train_data)


    def _get_dataset_class(self):
        return self.dataset_class

    
    # def _get_loss_func(self):
    #     return self.loss_fn


    def _init_model(self, train_data):
        self.frating = train_data.frating
        if self.fields is not None:
            assert self.frating in self.fields
            train_data.drop_feat(self.fields)
        else:
            self.fields = set(f for f in train_data.field2type if 'time' not in f)

        self.fiid = train_data.fiid
        assert self.fiid in self.fields
        
