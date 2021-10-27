import torchmetrics.functional as M

def recall(pred, count, k):
    output = pred[:, :k].sum(dim=-1).float() / count
    return output.mean()

metric_dict = {
    'ndcg': M.retrieval_normalized_dcg,
    'precision': M.retrieval_precision,
    'recall': recall,
    'map': M.retrieval_average_precision,
    #'hit': M.retrieval_hit_rate,
    'mrr': M.retrieval_reciprocal_rank,
    'rmse': M.mean_squared_error,
    'mse': M.mean_absolute_error,
    'auc': M.auroc,
    'logloss':None
}


def split_metrics(metric):
    topk_metrics = {'ndcg', 'precision', 'recall', 'map', 'mrr'}
    pred_metrics = {'rmse', 'mse', 'auc', 'logloss'}
    def f(pred, target, m):
        m_name = m[:m.index('@')].lower()
        if m_name in topk_metrics:
            topk = int(m[m.index('@')+1:])
            return metric_dict[m_name](pred, target, topk)
        else:
            return None
    rank_m = [(m, lambda *k: f(*k, m)) for m in metric]
    pred_m = [(m, metric_dict[m]) for m in metric if m in pred_metrics]
    return pred_m, rank_m