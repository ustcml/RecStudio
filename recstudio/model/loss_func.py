from operator import neg
import torch
import torch.nn.functional as F


class FullScoreLoss(torch.nn.Module):
    r"""Calculate loss with positive scores and scores on all items.

    The loss need user's perference scores on positive items(ground truth) and all other items.
    However, due to the item numbers are very huge in real-world datasets, calculating scores on all items
    may be very time-consuming. So the loss is seldom used in large-scale dataset.
    """

    def forward(self, label, pos_score, all_score, reduction='mean'):
        r"""
        """
        pass


class PairwiseLoss(torch.nn.Module):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        pass


class PointwiseLoss(torch.nn.Module):
    def forward(self, label, pos_score, reduction='mean'):
        raise NotImplementedError(f'{type(self).__name__} is an abstrat class, \
            this method would not be implemented')



class SoftmaxLoss(FullScoreLoss):
    def forward(self, label, pos_score, all_score, reduction='mean'):
        if all_score.dim() > pos_score.dim():
            loss = torch.logsumexp(all_score, dim=-1) - pos_score
        else:
            output = torch.logsumexp(all_score, dim=-1, keepdim=True) - pos_score
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
            loss = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
        
        if reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class BPRLoss(PairwiseLoss):
    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        if not self.dns:
            loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            loss = -loss * weight
            if reduction == 'mean':
                return torch.mean(loss.sum(-1))
        else:
            loss = -F.logsigmoid(pos_score - torch.max(neg_score, dim=-1))
            if reduction == 'mean':
                return torch.mean(loss)
        return loss


class Top1Loss(BPRLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        if not self.dns:
            loss = torch.sigmoid(neg_score - pos_score.view(*pos_score.shape, 1))
            loss += torch.sigmoid(neg_score ** 2)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            loss = (loss * weight).sum(-1)
        else:
            max_neg_score = torch.max(neg_score, dim=-1)
            loss = torch.sigmoid(max_neg_score-pos_score)
            loss = loss + torch.sigmoid(max_neg_score ** 2)
            
        if reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class SampledSoftmaxLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        if new_pos.dim() < new_neg.dim():
            new_pos.unsqueeze_(-1)
        new_neg = torch.cat([new_pos, new_neg], dim=-1)
        output = torch.logsumexp(new_neg, dim=-1, keepdim=True) - new_pos
        notpadnum = torch.logical_not(torch.isinf(new_pos)).float().sum(-1)
        loss = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
        
        if reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class WeightedBPRLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
        weight = F.softmax(neg_score - log_neg_prob, -1)
        loss = -(loss * weight).sum(-1)
        if reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class BinaryCrossEntropyLoss(PairwiseLoss):
    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        # pos_score: B | B x L | B x L
        # neg_score: B x neg | B x L x neg | B x neg
        assert ((pos_score.dim() == neg_score.dim()-1) and (pos_score.shape ==
                neg_score.shape[:-1])) or (pos_score.dim() == neg_score.dim())
        if not self.dns:
            weight = self._cal_weight(neg_score, log_neg_prob)
            padding_mask = torch.isinf(pos_score)
            # positive
            pos_loss = F.logsigmoid(pos_score)
            pos_loss.masked_fill_(padding_mask, 0.0)
            pos_loss = pos_loss.sum() / (~padding_mask).sum()
            # negative
            neg_loss = F.softplus(neg_score) * weight
            neg_loss = neg_loss.sum(-1)
            # mask
            if pos_score.dim() == neg_score.dim()-1:
                neg_loss.masked_fill_(padding_mask, 0.0)
                neg_loss = neg_loss.sum() / (~padding_mask).sum()
            else:
                neg_loss = torch.mean(neg_loss)

            return -pos_loss + neg_loss
        else:
            return torch.mean(-F.logsigmoid(pos_score) + F.softplus(torch.max(neg_score, dim=-1)))

    def _cal_weight(self, neg_score, log_neg_prob):
        return torch.ones_like(neg_score) / neg_score.size(-1)


class WeightedBinaryCrossEntropyLoss(BinaryCrossEntropyLoss):
    def _cal_weight(self, neg_score, log_neg_prob, reduction='mean'):
        return F.softmax(neg_score - log_neg_prob, -1)


class HingeLoss(PairwiseLoss):
    def __init__(self, margin=2, num_items=None):
        super().__init__()
        self.margin = margin
        self.n_items = num_items

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        loss = torch.maximum(torch.max(neg_score, dim=-1).values - pos_score +
                             self.margin, torch.tensor([0]).type_as(pos_score))
        if self.n_items is not None:
            impostors = neg_score - pos_score.view(-1, 1) + self.margin > 0
            rank = torch.mean(impostors, -1) * self.n_items
            loss = loss * torch.log(rank + 1)
        if reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class InfoNCELoss(SampledSoftmaxLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        return super().forward(label, pos_score, torch.zeros_like(pos_score),
                               neg_score, torch.zeros_like(neg_score), reduction)


class NCELoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        loss = -F.logsigmoid(new_pos) + (new_neg - F.softplus(new_neg)).sum(1)
        if reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class CCLLoss(PairwiseLoss):
    def __init__(self, margin=0.8, neg_weight=0.3) -> None:
        super().__init__()
        self.margin = margin
        self.neg_weight = neg_weight

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob, reduction='mean'):
        # pos_score: [B,] or [B, N]
        # neg_score: [B, num_neg] or [B, N, num_neg]
        pos_score = torch.sigmoid(pos_score)
        neg_score = torch.sigmoid(neg_score)
        neg_score_mean = torch.mean(torch.relu(neg_score - self.margin), dim=-1)  # [B] or [B,N]
        notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum()
        loss = (1 - pos_score) + self.neg_weight * neg_score_mean
        loss = torch.nan_to_num(loss, posinf=0.0)
        return loss.sum() / notpadnum


def l2_reg_loss_fn(*args):
    loss = 0.
    for emb in args:
        loss = loss + torch.mean(torch.sum(emb * emb, dim=-1)) # [B, D] -> [B] -> []
    return loss


class BCEWithLogitLoss(PointwiseLoss):
    def forward(self, label, pos_score, reduction='mean'):
        loss = F.binary_cross_entropy_with_logits(
            pos_score, label, reduction=reduction)
        return loss


class BCELoss(PointwiseLoss):
    def forward(self, label, pos_score, reduction='mean'):
        return F.binary_cross_entropy(pos_score, label, reduction=reduction)
    
    
class MSELoss(PointwiseLoss):
    def __init__(self, threshold: float=None) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, label, pos_score, reduction='mean'):
        if self.threshold is not None:
            label = (label > self.threshold).float()
        loss = F.mse_loss(pos_score, label, reduction=reduction)
        return loss
    
    
class L1Loss(PointwiseLoss):
    def __init__(self, threshold: float=None) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, label, pos_score, reduction='mean'):
        if self.threshold is not None:
            label = (label > self.threshold).float()
        loss = F.l1_loss(pos_score, label, reduction=reduction)
        return loss
    
    
class dCorLoss(PointwiseLoss):
    def forward(self, label, pos_score, reduction='mean'):
        """returns a value in [0, 1]"""
        assert reduction == 'mean', '`reduction` in dCorLoss must be `mean`.'
        pairwise_dis0 = torch.norm(label[:, None] - label, p = 2, dim = 2)
        pairwise_dis1 = torch.norm(pos_score[:, None] - pos_score, p = 2, dim = 2)

        center_dis_mat0 = pairwise_dis0 - pairwise_dis0.mean(dim=0)[None, :] - \
                            pairwise_dis0.mean(dim=1)[:, None] + pairwise_dis0.mean()
        center_dis_mat1 = pairwise_dis1 - pairwise_dis1.mean(dim=0)[None, :] - \
                            pairwise_dis1.mean(dim=1)[:, None] + pairwise_dis1.mean() 

        n = label.size(0)

        dcov2_01 = (center_dis_mat0 * center_dis_mat1).sum() / n**2
        dcov2_00 = (center_dis_mat0 * center_dis_mat0).sum() / n**2
        dcov2_11 = (center_dis_mat1 * center_dis_mat1).sum() / n**2
        dcor = -torch.sqrt(dcov2_01) / torch.sqrt(torch.sqrt(dcov2_00) * torch.sqrt(dcov2_11))
        return dcor