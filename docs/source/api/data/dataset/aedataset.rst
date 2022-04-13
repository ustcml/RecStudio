=====================================
torchrec.data.dataset.AEDataset 
=====================================

AEDataset is designed for Auto Encoder methods, such as `Multi_DAE`, `Multi_VAE`, `Rec_VAE`.

In auto encoder methods, the model input is usually a sequence of the user, which contains all the interations.

As for the splition method, usually the user's sequence is divided into 3 parts, for training, validation and test respectively.
In this case, the `split_mode` is `user_entry`. Another splition method is dividing all the users into training, validation and test part. 
However, in this case, `user_id` are required to not being used in model, because the user_id in validation and test parts couldn't be optimized.
The second `split_mode` is `user`.

To show a more clear understanding of `AEDataset`, here is an example in actual pratice to show what `__getitem__` method returns.




.. autoclass:: torchrec.data.dataset.AEDataset
    :members: