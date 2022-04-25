import torch
import torch.nn.functional as F
class FullScoreLoss(torch.nn.Module):
    r"""Calculate loss with positive scores and scores on all items.

    The loss need user's perference scores on positive items(ground truth) and all other items. 
    However, due to the item numbers are very huge in real-world datasets, calculating scores on all items
    may be very time-consuming. So the loss is seldom used in large-scale dataset.
    """
    def forward(self, label, pos_score, all_score):
        r"""
        """
        pass

class PairwiseLoss(torch.nn.Module):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        pass

class PointwiseLoss(torch.nn.Module):
    def forward(self, label, pos_score):
        raise NotImplementedError(f'{type(self).__name__} is an abstrat class, this method would not be implemented' )

class SquareLoss(PointwiseLoss):
    def forward(self, label, pos_score):
        if label.dim() > 1:
            return torch.mean(torch.mean(torch.square(label - pos_score), dim=-1))
        else:
            return torch.mean(torch.square(label - pos_score))

class SoftmaxLoss(FullScoreLoss):
    def forward(self, label, pos_score, all_score):
        if all_score.dim() > pos_score.dim():
            return torch.mean(torch.logsumexp(all_score, dim=-1) - pos_score)
        else:
            output = torch.logsumexp(all_score, dim=-1, keepdim=True) - pos_score
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
            output = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
            return torch.mean(output)

class BPRLoss(PairwiseLoss):
    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns:
            loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            return -torch.mean((loss * weight).sum(-1))
        else:
            loss = -torch.mean(F.logsigmoid(pos_score - torch.max(neg_score, dim=-1)))


class SampledSoftmaxLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        if new_pos.dim() < new_neg.dim():
            new_pos.unsqueeze_(-1)
        new_neg = torch.cat([new_pos, new_neg], dim=-1)
        output = torch.logsumexp(new_neg, dim=-1, keepdim=True) - new_pos
        notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
        output = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
        return torch.mean(output)

class WeightedBPRLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
        weight = F.softmax(neg_score - log_neg_prob, -1)
        return -torch.mean((loss * weight).sum(-1))

class BinaryCrossEntropyLoss(PairwiseLoss):
    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns or pos_score.dim() > 1:
            weight = F.softmax(torch.ones_like(neg_score), -1)
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
            output = torch.nan_to_num(F.logsigmoid(pos_score), nan=0.0).sum(-1) / notpadnum
            return torch.mean(-output + torch.sum(F.softplus(neg_score) * weight, dim=-1))
        else:
            return torch.mean(-F.logsigmoid(pos_score) + F.softplus(torch.max(neg_score, dim=-1)))
    
class WightedBinaryCrossEntropyLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        weight = F.softmax(neg_score - log_neg_prob, -1)
        if pos_score.dim() > 1:
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
            output = torch.nan_to_num(F.logsigmoid(pos_score), nan=0.0).sum(-1) / notpadnum
        else:
            output = F.logsigmoid(pos_score)
        return torch.mean(-output + torch.sum(F.softplus(neg_score) * weight, dim=-1))

class HingeLoss(PairwiseLoss):
    def __init__(self, margin=2, num_items=None):
        super().__init__()
        self.margin = margin
        self.n_items = num_items

    def forward(self, label, pos_score, log_pos_prob, neg_score, neg_prob):
        loss = torch.maximum(torch.max(neg_score, dim=-1).values - pos_score + self.margin, torch.tensor([0]).type_as(pos_score))
        if self.n_items is not None:
            impostors = neg_score - pos_score.view(-1, 1) + self.margin > 0
            rank = torch.mean(impostors, -1) * self.n_items
            return torch.mean(loss * torch.log(rank + 1))
        else:
            return torch.mean(loss)

class InfoNCELoss(SampledSoftmaxLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        return super().forward(label, pos_score, torch.zeros_like(pos_score), neg_score, torch.zeros_like(neg_score))


class NCELoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        loss = F.logsigmoid(new_pos) + (new_neg - F.softplus(new_neg)).sum(1)
        return -loss.mean()
