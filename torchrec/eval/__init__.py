import torchmetrics.functional as M
import torch.nn.functional as F
import torch
import sys
def recall(pred, target, k):
    count = (target > 0).sum(-1)
    output = pred[:, :k].sum(dim=-1).float() / count
    return output.mean()

def precision(pred, target, k):
    output = pred[:, :k].sum(dim=-1).float() / k
    return output.mean()

def map(pred, target, k):
    count = (target > 0).sum(-1)
    pred = pred[:, :k]
    output = pred.cumsum(dim=-1).float() / torch.arange(1, k+1, device=pred.device)
    output = (output * pred).sum(dim=-1) / torch.minimum(count, k*torch.ones_like(count))
    #torch.minimum(count, pred.sum(dim=-1).float())
    return output.mean()

def _dcg(pred, k):
    k = min(k, pred.size(1))
    denom = torch.log2(torch.arange(k, device=pred.device) + 2.0).view(1,-1)
    return (pred[:, :k] / denom).sum(dim=-1)

def ndcg(pred, target, k):
    pred_dcg = _dcg(pred, k)
    ideal_dcg = _dcg(torch.sort((target>0).float(), descending=True)[0], k) ## to do replace target>0 with target
    all_irrelevant = torch.all(target <= sys.float_info.epsilon, dim=-1)
    pred_dcg[all_irrelevant] = 0
    pred_dcg[~all_irrelevant] /= ideal_dcg[~all_irrelevant]
    return pred_dcg.mean()

def mrr(pred, target, k):
    row, col = torch.nonzero(pred[:, :k], as_tuple=True)
    row_uniq, inverse = torch.unique_consecutive(row, return_inverse=True)
    perm = torch.arange(inverse.size(0)).type_as(inverse)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    idx = inverse.new_empty(row_uniq.size(0)).scatter_(0, inverse, perm)
    first = row_uniq.new_zeros(pred.size(0)).scatter_(0, row_uniq, col[idx])
    output = 1.0 / (first + 1.0)
    output[first==0] = 0
    return output.mean()

def hits(pred, target, k):
    return torch.any(pred[:, :k] > 0, dim=-1).float().mean()

def logloss(pred, target):
    if pred.dim() == target.dim():
        return F.binary_cross_entropy_with_logits(pred, target.float())
    else:
        return F.cross_entropy(pred, target)

metric_dict = {
    'ndcg': ndcg,
    'precision': precision,
    'recall': recall,
    'map': map,
    'hit': hits,
    'mrr': mrr,
    'rmse': M.mean_squared_error,
    'mse': M.mean_absolute_error,
    'auc': M.auroc,
    'logloss':logloss
}


def get_rank_metrics(metric):
    if not isinstance(metric, list):
        metric = [metric]
    topk_metrics = {'ndcg', 'precision', 'recall', 'map', 'mrr', 'hit'}
    rank_m = [(m, metric_dict[m]) for m in metric if m in topk_metrics and m in metric_dict]
    return rank_m

def get_pred_metrics(metric):
    if not isinstance(metric, list):
        metric = [metric]
    pred_metrics = {'rmse', 'mse', 'auc', 'logloss'}
    pred_m = [(m, metric_dict[m]) for m in metric if m in pred_metrics and m in metric_dict]
    return pred_m