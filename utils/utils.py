from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

def dataset_collate(batch):
    elem = batch[0]
    batch_data = {key: pad_sequence([torch.from_numpy(d[key]) for d in batch], batch_first=True) \
        if isinstance(elem[key], np.ndarray) else [d[key] for d in batch] for key in elem}
    return batch_data