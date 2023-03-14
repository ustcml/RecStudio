import os, sys
sys.path.append(".")
# sys.path.append(os.path.join(__file__, '../'))
# sys.path.insert(0, os.path.join(__file__, '../'))

from recstudio.data.dataset import TripletDataset

data = TripletDataset(name='ml-100k')
trn, val, tst = data.build(split_ratio=[0.7, 0.2, 0.1])

trn_loader = trn.train_loader(batch_size=128, shuffle=True)

batch = next(iter(trn_loader))
print(batch)

# print("End.")