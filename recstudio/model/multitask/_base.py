import torch
from recstudio.data.dataset import TripletDataset
from recstudio.model.basemodel import BaseRanker
from recstudio.model.loss_func import BCEWithLogitLoss


class _MultiTaskBase(BaseRanker):

    def _get_dataset_class():
        return TripletDataset
    
    def _get_loss_func(self):
        return BCEWithLogitLoss()


    def training_step(self, batch):
        y_h, _ = self.forward(batch)
        weights = self.config['train'].get('multitask_weights', None)
        if weights is None:
            weights = [1.0] * len(self.frating)
        assert len(weights) == len(self.frating), \
            f'Expect {len(self.frating)} float(s) for weights, but got {self.config["train"]["weights"]} with length {len(weights)}.'
        weights = torch.tensor(weights, device=self.device)

        loss = {}
        loss['loss'] = 0.0
        for i, r in enumerate(self.frating):
            loss[r] = self.loss_fn(**y_h[r])
            loss['loss'] = loss['loss'] + weights[i] * loss[r]
 
        return loss
