from typing import Any, Dict, Iterator, List, Optional, Union
from torchrec.utils.utils import set_color, parser_yaml, color_dict, print_logger
from pytorch_lightning.loops import FitLoop
from pytorch_lightning.loops import TrainingBatchLoop, TrainingEpochLoop
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
import torch


class KGFitLoop(FitLoop):
    def __init__(self, min_epochs: Optional[int] = None, max_epochs: Optional[int] = None):
        super().__init__(min_epochs, max_epochs)
        self.epoch_loop = KGTrainEpochLoop()

    def get_train_dataloader(self):
        train_dataloader = self.trainer.accelerator.process_dataloader(self.trainer.train_dataloader)
        train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)        
        return train_dataloader

    def run_one_epoch(self, val_flag):
        train_dataloader = self.get_train_dataloader()
        self.epoch_loop.val_flag = val_flag
        self.trainer.logger_connector._logged_metrics = {'epoch' : self.current_epoch}
        epoch_output = self.epoch_loop.run(train_dataloader)

        self.global_step -= 1
        # log epoch metrics
        self.trainer.logger_connector.update_train_epoch_metrics()
        self.global_step += 1


class KGTrainEpochLoop(TrainingEpochLoop):
    def __init__(self) -> None:
        super().__init__(None, None)
        self.batch_loop = TrainingBatchLoop()
        self.val_loop = EvaluationLoop()
        self.val_flag = True

    def on_advance_end(self) -> None:
        if self.val_flag:
            super().on_advance_end()

class MKRFitLoop(KGFitLoop):
    
    def __init__(self, kge_interval, max_epochs: Optional[int] = 1000) -> None:
        super().__init__(max_epochs=max_epochs)
        self.kge_interval = kge_interval
    
    def advance(self, *args, **kwargs) -> None:
        """Runs one whole epoch."""
        with self.trainer.profiler.profile("run_training_epoch"):
            if (self.current_epoch + 1) % self.kge_interval == 0:
                # run train Rec epoch
                self.run_one_epoch(False)
                print_logger.info(
                    set_color(
                        'Rec epoch finished. It is followed by a kg epoch and validation will be done in the kg epoch.', 
                        'yellow'
                    )
                )
                # run train Kg epoch
                self.run_one_epoch(True)
            else: 
                # run train Rec epoch
                self.run_one_epoch(True)
   
class MKRTrainEpochLoop(TrainingEpochLoop):
    
    def __init__(self) -> None:
        super().__init__(None, None)
        self.batch_loop = TrainingBatchLoop()
        self.val_loop = EvaluationLoop()
        self.run_val = True

    def on_advance_end(self) -> None:
        if self.run_val:
            super().on_advance_end()

class KGATFitLoop(KGFitLoop):
    
    def __init__(self, max_epochs: Optional[int] = 1000) -> None:
        super().__init__(max_epochs=max_epochs)
        
    def advance(self, *args, **kwargs) -> None:
        """Runs one whole epoch."""
        with self.trainer.profiler.profile("run_training_epoch"):
            # run train Rec epoch
            self.run_one_epoch(False)
            # run train Kg epoch
            self.run_one_epoch(True)