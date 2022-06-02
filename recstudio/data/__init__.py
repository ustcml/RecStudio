from recstudio.data.dataset import MFDataset, SeqDataset, AEDataset, FullSeqDataset
from recstudio.data.advance_dataset import ALSDataset

import os

supported_dataset = []
for f in os.listdir(os.path.join(os.path.dirname(__file__), 'config')):
    if f != "all.yaml":
        supported_dataset.append(f.split(".")[0])
