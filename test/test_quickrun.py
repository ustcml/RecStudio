import sys
sys.path.append(".")
from recstudio import quickstart

quickstart.run(model='MultiVAE', dataset='ml-100k', gpu=[2])


import recstudio.data as recdata

print(recdata.supported_dataset)