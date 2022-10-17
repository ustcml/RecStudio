import recstudio.eval as eval
import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, scorer

# TODO: out-of-memory problem when predict for large dataset

r"""
DIN
######################

Paper Reference:
    Guorui Zhou, et al. "Deep Interest Network for Click-Through Rate Prediction" in KDD2018.
    https://dl.acm.org/doi/10.1145/3219819.3219823

"""


class DIN(basemodel.BaseRanker):
    r"""
        | Deep Interest Network (DIN) designs a local activation unit to adaptively learn the representation
          of user interests from historical behaviors with respect to a certain ad.

        | DIN calculate the relevance between the target item and items in the sequence by adapting an
          attention machnism. But the method could not be applied to recall on all items in prediction
          due to the huge time cost.
    """
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('DIN')
        parent_parser.add_argument("--activation", type=str, default='dice', help='activation for MLP')
        parent_parser.add_argument("--attention_mlp", type=int, nargs='+', default=[128, 64], help='MLP layer size for attention calculation')
        parent_parser.add_argument("--fc_mlp", type=int, nargs='+', default=[128, 64, 64], help='MLP layer size for the MLP before prediction')
        parent_parser.add_argument("--negative_count", type=int, default=1, help='negative sampling numbers')
        parent_parser.add_argument("--dropout", type=float, default=0.3, help='dropout rate for MLP')
        return parent_parser


    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.sampler = sampler.MaskedUniformSampler(train_data.num_items)


    def _set_data_field(self, data):
        pass


    def _get_dataset_class():
        r"""The dataset is SeqDataset."""
        return dataset.SeqDataset


    def _get_retriever(self, train_data):
        return sampler.UniformSampler(train_data.num_items)


    def _get_scorer(self, train_data):
        return module.ctr.DINScorer(
            train_data.fuid, train_data.fiid, train_data.num_users, train_data.num_items,
            self.embed_dim, self.config['attention_mlp'], self.config['fc_mlp'], dropout=self.config['dropout'],
            activation=self.config['activation'], batch_norm=False
        )


    def _get_loss_func(self):
        r"""BinaryCrossEntropy is used as the loss function."""
        return loss_func.BinaryCrossEntropyLoss()

    @torch.no_grad()
    def topk(self, batch, k, user_h):
        num_items = self.sampler.num_items
        step = 40
        device = batch[self.fiid].device
        bs = batch[self.fiid].size(0)
        all_score = []
        for s in range(1, num_items+1, step):
            all_id = torch.arange(s, min(num_items+1, s+step), device=device, dtype=torch.long).squeeze(0).repeat(bs, 1)
            all_item_batch = self._generate_neg_batch(batch, all_id)
            score = self.scorer(all_item_batch).view(-1, all_id.size(1))
            all_score.append(score)

        more = user_h.size(1) if user_h is not None else 0
        all_score = torch.cat(all_score, dim=-1)
        score, topk_items = torch.topk(all_score, k+more)
        topk_items += 1
        if user_h is not None:
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score, idx = score.topk(k)
            topk_items = torch.gather(topk_items, 1, idx)
        return score, topk_items



    @torch.no_grad()
    def _test_step(self, batch, metric, cutoffs=None):
        rank_m = eval.get_rank_metrics(metric)
        pred_m = eval.get_pred_metrics(metric)
        bs = batch[self.frating].size(0)
        if len(rank_m) > 0:
            assert cutoffs is not None, 'expected cutoffs for topk ranking metrics.'
            topk = self.config['topk']
            score, topk_items = self.topk(batch, topk, batch['user_hist'])
            if batch[self.fiid].dim() > 1:
                target, _ = batch[self.fiid].sort()
                idx_ = torch.searchsorted(target, topk_items)
                idx_[idx_ == target.size(1)] = target.size(1) - 1
                target = torch.gather(target, 1, idx_)
                label = target == topk_items
                pos_rating = batch[self.frating]
            else:
                target = batch[self.fiid].view(-1, 1)
                label = target == topk_items
                pos_rating = batch[self.frating].view(-1, 1)

            metric = {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}
        else:
            metric = {}


        # calculate AUC and logloss
        test_neg_id, _ = self.sampler(batch[self.frating].new_zeros((bs, 1)), 1, user_hist=batch['user_hist'])
        pos_score_ra = self.scorer(batch)
        test_neg_batch = self._generate_neg_batch(batch, test_neg_id)
        test_neg_score_ra = self.scorer(test_neg_batch)
        pred = torch.cat((pos_score_ra, test_neg_score_ra), dim=0)
        target = torch.cat((torch.ones_like(pos_score_ra), torch.zeros_like(test_neg_score_ra)), dim=0)
        auc = torch.mean((pos_score_ra > test_neg_score_ra).float())
        logloss = eval.logloss(pred, target)
        metric['auc'] = auc
        metric['logloss'] = logloss
        return metric, bs
