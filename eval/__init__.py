import torchmetrics.functional as M

metric_dict = {
    'ndcg': M.retrieval_normalized_dcg,
    'precision': M.retrieval_precision,
    'recall': M.retrieval_recall,
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
    rank_m = [(m, lambda *k: metric_dict[m](k[0], k[1], int(m[m.index('@')+1:]))) for m in metric if m[:m.index('@')] in topk_metrics]
    pred_m = [(m, metric_dict[m]) for m in metric if m in pred_metrics]
    return pred_m, rank_m