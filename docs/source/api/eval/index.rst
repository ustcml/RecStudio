====================================
recstudio.eval
====================================

RecStudio provides a series of functions for evaluation, which are suitable for GPU tensors with PyTorch.
With those metric functions, the result predicted by models can be directly used to calculate metrics like 
`NDCG`, `Recall`, `Precision`, `MRR`, `mAP`, `logloss`, `Hits` and so on.

In order to make those function easier to use, a unified input paramerters are designed for all metrics: 

- `pred`: [B, num_items]. The prediction result of the model with bool type values. If the value in the j-th column 
    is `True`, the j-th highest item predicted by model is right.

- `target`: [B, num_target]. The ground truth. In different tasks, the target is different. In general recommendation, the target is 
    a subset of user's interacted items. In sequential recommendation, the target is the next item of the user interacted item sequence.

- `k`: `int`. Calculating metric on the most k relavant items. Usually, the metric will be represents as `m@k`, like `recall@10`, `NDCG@50`.

.. autofunction:: recstudio.eval.precision
.. autofunction:: recstudio.eval.recall
.. autofunction:: recstudio.eval.ndcg
.. autofunction:: recstudio.eval.mrr
.. autofunction:: recstudio.eval.map
.. autofunction:: recstudio.eval.hits
.. autofunction:: recstudio.eval.logloss