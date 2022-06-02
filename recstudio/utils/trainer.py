import os, torch, logging
import numpy as np
from typing import List, Union, Tuple, Dict, Optional
from recstudio.utils.utils import print_logger

# TODO: merge trainer into basemodel

class Trainer():
    
    def __init__(
        self,
        max_epochs: int,
        gpus: Union[int, List[int]],
        num_sanity_val_steps: int,
        progress_bar_refresh_rate: Optional[int] = 0,
        multiple_trainloader_mode: Optional[Union[str, None]] = None,
        logger: Optional[Union[bool, logging.Logger]] = None,
        accelerator: Optional[str] = 'cpu',
        strategy: Optional[Union[str, None]] = None,
        validtion_on_start: Optional[bool] = False
        ) -> None:
        
        self.nepoch = 0
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.num_sanity_val_step = num_sanity_val_steps
        self.progress_bar_refresh_rate = progress_bar_refresh_rate

        if multiple_trainloader_mode in ['min_size', 'max_size']:
            self.multiple_trainloader_mode = multiple_trainloader_mode
        else:
            raise ValueError(f"`multiple_trainloader_mode` can only be \
                `min_size` or `max_size`, but `{multiple_trainloader_mode}` \
                is given.")

        self.logger = logger

        if accelerator in ['cpu', 'gpu']:
            self.accelerator = accelerator
        else:
            raise ValueError(f"`accelerator` can only be `cpu` or `gpu`, \
                but `{accelerator}` is given.")

        if strategy in ['dp', 'ddp', None]:
            self.strategy = strategy
        else:
            raise ValueError(f"`strategy` can only be `dp` or `ddp` or None, \
                but `{strategy}` is given.")

        self.validation_on_start = validtion_on_start


    def fit(
        self, 
        model: torch.nn.Module, 
        # trn_dataloader: torch.utils.data.DataLoader, 
        val_dataloader: torch.utils.data.DataLoader
        ):

        try:
            # validation_metrics = {}
            if (self.validation_on_start) and (val_dataloader is not None):
                validation_output_list = self.validation_epoch(val_dataloader)
                model.validation_epoch_end(validation_output_list)

            for e in range(self.max_epochs):
                # training procedure
                training_output_list = self.training_epoch()

                # validation procedure
                if val_dataloader is not None:
                    validation_output_list = self.validation_epoch(val_dataloader)
                    model.validation_epoch_end(validation_output_list)

                model.training_epoch_end(training_output_list)
                self.nepoch += 1

                # learning rate scheduler step
                for opt in model.current_epoch_optimizers:
                    if 'scheduler' in opt:
                        opt['scheduler'].step()

                if model.callback is not None:
                    stop_training_sig = \
                        model.callback(model, e, model.logged_metrics)

                    if stop_training_sig:
                        break
            
            # testing procedure
            model.callback.save_checkpoint(model) #TODO: case when callback is none
        
        except KeyboardInterrupt:
            # if catch keyboardinterrupt in training, save the best model.
            model.callback.save_checkpoint(model)

 

    def training_epoch(self, model):
        model.train()
        output_list = []
        for opt in model.current_epoch_optimizers:
            opt['optimizer'].zero_grad()

        trn_dataloaders, combine = model.current_epoch_trainloaders
        if isinstance(trn_dataloaders, list) or isinstance(trn_dataloaders, Tuple):
            if combine:
                trn_dataloaders = [CombinedLoaders(list(trn_dataloaders))]
        else:
            trn_dataloaders = [trn_dataloaders]

        for loader in trn_dataloaders:
            for batch in loader:
                # data to device
                batch = self._batchdata_to_device(batch)

                # model loss
                loss = model.training_step(batch)

                loss.backward()

                for opt in model.current_epoch_optimizers:
                    opt['optimizer'].step()
                
                output_list.append(loss)

        return output_list


    def validation_epoch(self, model, dataloader):
        model.eval()
        output_list = []

        for batch in dataloader:
            # data to device
            batch = self._batchdata_to_device(batch)

            # model validation results
            output = model.validation_step(batch)
            
            output_list.append(output)
        
        return output_list


    def test_epoch(self, model, dataloader):
        model.eval()
        output_list = []

        for batch in dataloader:
            # data to device
            batch = self._batchdata_to_device(batch)

            # model validation results
            output = model.test_step(batch)
            
            output_list.append(output)
        
        return output_list


    def test(self, model, dataloader):
        metrics = {}
        model.callback.load_checkpoint(model, model.callback.save_path)
        output_list = self.test_epoch(model, dataloader)
        metrics.update(
            model.test_epoch_end(output_list)
        )
        return metrics


    def _batchdata_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.cuda() if self.accelerator=='gpu' else batch
        elif isinstance(batch, Dict):
            for k in batch:
                batch[k] = self._batchdata_to_device(batch[k])
                return batch
        elif isinstance(batch, List) or isinstance(batch, Tuple):
            output = []
            for b in batch:
                output.append(self._batchdata_to_device(b))
            return output if isinstance(batch, List) else tuple(output)
        else:
            raise TypeError(f"`batch` is expected to be torch.Tensor, Dict, \
                List or Tuple, but {type(batch)} given.")

        
class EarlyStopping(object):
    def __init__(
        self,
        monitor: str,
        save_dir: Optional[str] = None,
        filename: Optional[str] = None,
        patience: Optional[int] = 10,
        delta: Optional[float] = 0,
        mode: Optional[str] = 'max',
        ):
        r"""
        Early Stop and Model Checkpoint save callback.

        Args:

            monitor: quantity to monitor. By default it is None 
                which saves a checkpoint only for the last epoch.

            save_dir: directory to save checkpoint. By default it is None 
                which means not saving checkpoint.

            filename: filename of the checkpoint file. By default it is 
                None which will be set as "epoch={}-val_{}={}.ckpt"

            patience: number of checks with no improvement after which training 
                will be stopped. One check happens after every training epoch.

            delta: minimum change in the monitored quantity to qualify as an 
                improvement, i.e. an absolute change of less than or equal to 
                `min_delta`, will count as no improvement.

            mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training will 
                stop when the quantity monitored has stopped decreasing and 
                in ``'max'`` mode it will stop when the quantity monitored has 
                stopped increasing.
            
        """

        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        
        self.save_dir = save_dir
        self.filename = filename
        self.__check_save_dir()

        if mode in ['min', 'max']:
            self.mode = mode
        else:
            raise ValueError(f"`mode` can only be `min` or `max`, \
                but `{mode}` is given.")
        
        self.__counter = 0
        self.best_value = np.inf if self.mode=='min' else -np.inf
        self.best_value_epoch = 0


    def __check_save_dir(self):
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)


    def __call__(self, model, epoch, metrics):
        if self.monitor not in metrics:
            raise ValueError(f"monitor {self.monitor} not in given `metrics`.")

        if self.mode == 'max':
            if metrics[self.monitor] >= self.best_value+self.delta:
                self.__reset_counter(epoch, metrics[self.monitor])
                self.save_checkpoint(model)
            else:
                self.__counter += 1
        else:
            if metrics[self.monitor] <= self.best_value-self.delta:
                self.__reset_counter(epoch, metrics[self.monitor])
                self.save_checkpoint(model)
            else:
                self.__counter += 1
        
        if self.__counter >= self.patience:
            print_logger.info(f"early stoped since the metric {self.monitor} \
                haven't been improved for {self.counter} epochs.")
            print_logger.info(f"the best score of {self.monitor} is \
                {self.best_value} on epoch {self.best_value_epoch}")
            return True
        else:
            return False


    def __reset_counter(self, epoch, value):
        self.__counter = 0
        self.best_value = value
        self.best_value_epoch = epoch

    
    def save_checkpoint(self, model):
        if self.save_dir is not None:
            ckpt = {}
            ckpt['config'] = model.config
            ckpt['parameters'] = model.state_dict()
            if self.filename is not None:
                filename = self.filename
            else:
                filename = "epoch={}-val_{}={:.3d}.ckpt".format(
                    self.best_value_epoch, self.monitor, self.best_value)

            self.save_path = os.path.join(self.save_dir, filename)
            torch.save(ckpt, self.save_path)
            print_logger(f"best model checkpoint saved in {self.save_path}.")

    
    def load_checkpoint(self, model, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        else:
            ckpt = torch.load(path)
            model.config = ckpt['config']
            model.load_state_dict(ckpt['parameters'])
            # model = 



class CombinedLoaders(object):
    def __init__(self, loaders) -> None:
        r"""
        The first loader is the main loader.
        """
        self.loaders = loaders


    def __len__(self):
        return len(self.loaders[0])


    def __iter__(self):
        for i, l in enumerate(self.loaders):
            self.loaders[i] = iter(l)
        return self

    
    def __next__(self):
        batch = next(self.loaders[0])
        for i, l in enumerate(self.loaders[1:]):
            try:
                batch.update(next(l))
            except StopIteration:
                self.loaders[i+1] = iter(self.loaders[i+1])
                batch.update(next(self.loaders[i+1]))
        return batch
        
