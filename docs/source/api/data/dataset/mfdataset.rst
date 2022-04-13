=================================
recstudio.data.dataset.MFDataset
=================================

MFDataset is the basic dataset in RecStudio, which is designed mainly for 
Matric Factorization methods. The dataset returns `<u, i>` pairs, where user 
`u` interacts with `i`. 

.. autoclass:: recstudio.data.dataset.MFDataset
    :members:

    .. automethod:: __init__(self, config_path:str)
    .. automethod:: __len__(self)
    .. automethod:: __getitem__(self, index:int)

