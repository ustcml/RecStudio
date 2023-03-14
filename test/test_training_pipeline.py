import os
import sys
from tqdm import tqdm
from termcolor import colored


dir = os.path.join( os.path.dirname(__file__), '../' )
sys.path.append( os.path.abspath(dir) )

from recstudio.quickstart import run

# Note: please add models to be tested here
datasets = ['ml-100k']

fm_model = ['DCN', 'DeepFM', 'FM', 'LR', 'NFM', 'WideDeep', 'xDeepFM']
mf_model = ['BPR', 'CML', 'DSSM', 'EASE', 'IRGAN', 'ItemKNN', 'LogisticMF', 'NCF', 'SLIM', 'WRMF']
seq_model = ['BERT4Rec', 'Caser', 'FPMC', 'GRU4Rec', 'HGN', 'NARM', 'NPE', 'SASRec', 'STAMP', 'TransRec']
ae_model = ['MultiDAE', 'MultiVAE']
graph_model = ['LightGCN', 'NCL', 'NGCF', 'SGL', 'SimGCL']
cl_seq_model = ['CL4SRec', 'CoSeRec', 'ICLRec']


training_configs = [
    {'train': {'epochs': 3} },
]

all_models = {
    'FM': fm_model,
    'MF': mf_model,
    'SEQ': seq_model,
    'AE': ae_model,
    'GRAPH': graph_model,
    'CL-SEQ': cl_seq_model,
}

# test loop
num_exps = sum([len(m) for m in all_models.values()]) * len(datasets) * len(training_configs)
pbar = tqdm(total=num_exps)
for cate, models in all_models.items():
    tqdm.write(f"Test {cate} models - {len(mf_model)} models:")
    failed_exp = []
    for m in models:
        for d in datasets:
            for i, config in enumerate(training_configs):
                pbar.update(1)
                tqdm.write(colored(f"### Test: model-{m}, data-{d}, {i}-th configurations.", on_color='on_blue'))
                try:
                    run(m, d, config, verbose=False)
                    tqdm.write(colored(f"$$$ Test passed!", 'green'))
                except:
                    tqdm.write(colored(f"!!! Test failed!", 'red'))
                    failed_exp.append({
                        'model': m,
                        'dataset': d,
                        'config': config
                    })
    tqdm.write("{} models test End. {}/{} failed.".format( cate, len(failed_exp), (len(models) * len(datasets) * len(config)) ))

pbar.close()