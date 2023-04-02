import torch
from .debiasedretriever import DebiasedRetriever
from recstudio.data import DICEDataset
from recstudio.model import basemodel

r"""
DICE
#########

Paper Reference:
    Disentangling User Interest and Conformity for Recommendation with Causal Embedding (WWW'21)
    https://doi.org/10.1145/3442381.3449788
"""

class DICE(DebiasedRetriever):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('DICE')
        parent_parser.add_argument("--discrepancy", type=str, default='l1', help='discrepency loss function')
        parent_parser.add_argument("--dis_penalty", type=float, default=0.01, help='discrepency penalty')
        parent_parser.add_argument("--int_weight", type=float, default=0.1, help='weight for interest term in the loss function')
        parent_parser.add_argument("--con_weight", type=float, default=0.1, help='weight for popularity term in the loss function')
        parent_parser.add_argument("--margin_up", type=float, default=40.0, help='margin for negative but more popular sampling')
        parent_parser.add_argument("--margin_down", type=float, default=40.0, help='margin for negative and less popular sampling')
        parent_parser.add_argument("--pool", type=int, default=40, help='pool for negative sampling')
        parent_parser.add_argument("--adaptive", type=bool, default=True, help='adapt hyper-parameters or not')
        parent_parser.add_argument("--margin_decay", type=float, default=0.9, help='decay of margin')
        parent_parser.add_argument("--loss_decay", type=float, default=0.9, help='decay of loss')
        return parent_parser  
        
    def _get_dataset_class():
        return DICEDataset 

    def _get_final_loss(self, loss : dict, output : dict, batch : dict):
        query_int = output['interest']['query']
        query_con = output['conformity']['query']
        pos_item_int = output['interest']['item']
        pos_item_con = output['conformity']['item']
        neg_item_int = output['interest']['neg_item']
        neg_item_con = output['conformity']['neg_item']
        item_int = torch.vstack((pos_item_int, neg_item_int.view(-1, pos_item_int.shape[1])))
        item_con = torch.vstack((pos_item_con, neg_item_con.view(-1, pos_item_con.shape[1])))
        loss_dis = self.discrepancy(query_int, query_con) + self.discrepancy(item_int, item_con)
        loss_click = self.backbone['interest'].loss_fn(
            pos_score=output['interest']['score']['pos_score'] + output['conformity']['score']['pos_score'], 
            neg_score=output['interest']['score']['neg_score'] + output['conformity']['score']['neg_score'], 
            label=None, log_pos_prob=None, log_neg_prob=None)

        mask = batch['mask']
        loss_int = torch.mean(mask * loss['interest'])
        loss_con = torch.mean(~mask * loss['conformity']) + \
                    torch.mean(mask * self.backbone['conformity'].loss_fn(
                        pos_score=output['conformity']['score']['neg_score'], 
                        neg_score=output['conformity']['score']['pos_score'],
                        label=None, log_pos_prob=None, log_neg_prob=None
                    ))

        return self.int_weight * loss_int + self.con_weight * loss_con + \
                loss_click - self.config['train']['dis_penalty'] * loss_dis
    
    def _adapt(self, current_epoch):
        if not hasattr(self, 'last_epoch'):
            self.last_epoch = 0
            self.int_weight = self.config['train']['int_weight']
            self.con_weight = self.config['train']['con_weight']
        if current_epoch > self.last_epoch:
            self.last_epoch = current_epoch
            self.int_weight = self.int_weight * self.config['train']['loss_decay']
            self.con_weight = self.con_weight * self.config['train']['loss_decay']

    def training_step(self, batch, nepoch):
        self._adapt(nepoch)
        return super().training_step(batch, nepoch)
        
    def forward(self, batch):
        query = self.query_encoder(self._get_query_feat(batch))
        query = self.query_encoder.split(query)
        neg_item_idx = batch['neg_items']
        output = {}
        for name, backbone in self.backbone.items():
            output[name] = backbone.forward(
                batch, 
                False,
                return_query=True, 
                return_item=True,
                return_neg_item=True,
                return_neg_id=True,
                query=query[name], 
                neg_item_idx=neg_item_idx)
        return output    