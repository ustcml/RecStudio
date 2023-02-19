import os
import numpy as np
import torch
import copy
import logging
from typing import List, Union, Tuple, Dict, Optional


class EarlyStopping(object):
    def __init__(
        self,
        model: torch.nn.Module,
        monitor: str,
        dataset_name: str,
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

        self.model_name = model.__class__.__name__
        self.save_dir = save_dir
        self.filename = filename
        self.__check_save_dir()

        if mode in ['min', 'max']:
            self.mode = mode
        else:
            raise ValueError(f"`mode` can only be `min` or `max`, \
                but `{mode}` is given.")

        self._counter = 0
        self.best_value = np.inf if self.mode=='min' else -np.inf
        self.logger = logging.getLogger('recstudio')

        self.best_ckpt = {
            'config': model.config,
            'model': self.model_name,
            'epoch': 0,
            'parameters': copy.deepcopy(model._get_ckpt_param()),
            'metric': {self.monitor: np.inf if self.mode=='min' else -np.inf}
        }

        if filename != None:
            self._best_ckpt_path = filename
        else:
            _file_name = None 
            for handler in self.logger.handlers:
                if type(handler) == logging.FileHandler:
                    _file_name = os.path.basename(handler.baseFilename).split('.')[0]
            if _file_name is None:
                import time
                _file_name = time.strftime(f"{self.model_name}-{dataset_name}-%Y-%m-%d-%H-%M-%S.log", time.localtime())
            self._best_ckpt_path = f"{_file_name}.ckpt"

    def __check_save_dir(self):
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def __call__(self, model, epoch, metrics):
        if self.monitor not in metrics:
            raise ValueError(f"monitor {self.monitor} not in given `metrics`.")
        if self.mode == 'max':
            if metrics[self.monitor] >= self.best_value+self.delta:
                self._reset_counter(model, epoch, metrics)
                self.logger.info("{} improved. Best value: {:.4f}".format(
                                self.monitor, metrics[self.monitor]))
            else:
                self._counter += 1
        else:
            if metrics[self.monitor] <= self.best_value-self.delta:
                self._reset_counter(model, epoch, metrics[self.monitor])
                self.logger.info("{} improved. Best value: {:.4f}".format(
                                self.monitor, metrics[self.monitor]))
            else:
                self._counter += 1

        if self._counter >= self.patience:
            self.logger.info(f"Early stopped. Since the metric {self.monitor} "
                             f"haven't been improved for {self._counter} epochs.")
            self.logger.info(f"The best score of {self.monitor} is "
                             f"{self.best_value:.4f} on epoch {self.best_ckpt['epoch']}")
            return True
        else:
            return False

    def _reset_counter(self, model: torch.nn.Module, epoch, value):
        self._counter = 0
        self.best_value = value[self.monitor]
        self.best_ckpt['parameters'] = copy.deepcopy(model._get_ckpt_param())
        self.best_ckpt['metric'] = value
        self.best_ckpt['epoch'] = epoch

    def save_checkpoint(self, epoch): # TODO haddle saving checkpoint in ddp
        if self.save_dir is not None:
            self.save_path = os.path.join(self.save_dir, self._best_ckpt_path)
            torch.save(self.best_ckpt, self.save_path)
            self.logger.info(f"Best model checkpoint saved in {self.save_path}.")
        else:
            raise ValueError(f"fail to save the model, self.save_dir can't be None!")

    def get_checkpoint_path(self):
        return self._best_ckpt_path


class SaveLastCallback(object):

    def __init__(
        self,
        model:torch.nn.Module,
        dataset_name: str,
        save_dir: Optional[str] = None,
        filename: Optional[str] = None
        ):
        self.model_name = model.__class__.__name__
        self.logger = logging.getLogger('recstudio')
        self.last_ckpt = {
            'config': model.config,
            'model': self.model_name,
            'epoch': 0,
            'parameters': copy.deepcopy(model._get_ckpt_param()),
            'metrics' : None
        }
        self.save_dir = save_dir
        self.__check_save_dir()

        if filename != None:
            self._last_ckpt_path = filename
        else:
            _file_name = None 
            for handler in self.logger.handlers:
                if type(handler) == logging.FileHandler:
                    _file_name = os.path.basename(handler.baseFilename).split('.')[0]
            if _file_name is None:
                import time
                _file_name = time.strftime(f"{self.model_name}-{dataset_name}-%Y-%m-%d-%H-%M-%S.log", time.localtime())
            self._last_ckpt_path = f"{_file_name}.ckpt"

    def __check_save_dir(self):
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def __call__(self, model:torch.nn.Module, epoch, metrics):
        self.last_ckpt['epoch'] = epoch
        self.last_ckpt['metrics'] = metrics
        self.last_ckpt['parameters'] = copy.deepcopy(model._get_ckpt_param())
        return False

    def save_checkpoint(self, epoch):
        self.save_path = os.path.join(self.save_dir, self._last_ckpt_path)
        torch.save(self.last_ckpt, self.save_path)
        self.logger.info(f"Last model is saved in {self.save_path}.")

    def get_checkpoint_path(self):
        return self._last_ckpt_path


class IntervalCallback(object):

    def __init__(self,
        model:torch.nn.Module,
        print_logger,
        dataset_name: str,
        save_dir: Optional[str] = None,
        filename: Optional[str] = None,
        interval_epochs:int = 20
        ) -> None:
        self.interval_epochs = interval_epochs
        self.model_name = model.__class__.__name__
        self.logger = print_logger
        self.interval_ckpt = {
            'config': model.config,
            'model': self.model_name,
            'epoch': 0,
            'parameters': model._get_ckpt_param(),
            'metrics': None
        }
        self.save_dir = save_dir
        self.__check_save_dir()

        if filename != None:
            self._interval_ckpt_path = filename
        else:
            _file_name = None 
            for handler in self.logger.handlers:
                if type(handler) == logging.FileHandler:
                    _file_name = os.path.basename(handler.baseFilename).split('.')[0]
            if _file_name is None:
                import time
                _file_name = time.strftime(f"{self.model_name}-{dataset_name}-%Y-%m-%d-%H-%M-%S.log", time.localtime())
            self._interval_ckpt_path = f"{_file_name}.ckpt"

        self.current_epoch = 0

    def __check_save_dir(self):
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def __call__(self, model:torch.nn.Module, epoch, metrics):
        if (epoch + 1) % self.interval_epochs == 0:
            self.interval_ckpt['epoch'] = epoch
            self.interval_ckpt['metrics'] = metrics
            self.interval_ckpt['parameters'] = copy.deepcopy(model._get_ckpt_param())
            self.save_checkpoint(epoch)
        return False

    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.save_dir, f"{epoch + 1}_epochs-{self._interval_ckpt_path}.ckpt")
        self.current_epoch = epoch + 1
        torch.save(self.interval_ckpt, save_path)
        self.logger.info(f"Model at epoch {epoch + 1} is saved in {save_path}.")

    def get_checkpoint_path(self, nepoch=None):
        if nepoch == None:
            if self.current_epoch == 0:
                return None 
            else:
                return os.path.join(self.save_dir, f"{self.current_epoch}_epochs-{self._interval_ckpt_path}.ckpt")
        else:
            assert nepoch <= self.current_epoch and nepoch % self.interval_epochs == 0
            return os.path.join(self.save_dir, f"{nepoch}_epochs-{self._interval_ckpt_path}.ckpt")
