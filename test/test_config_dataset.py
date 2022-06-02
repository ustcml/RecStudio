import os, sys
sys.path.append(".")

from recstudio.quickstart import generate_dataset_config

generate_dataset_config(name='mydataset', data_dir='dataset_dir/', 
    interaction_file='inter.csv', user_id='user_id', item_id='item_id', 
    rating='rating', timestamp='timestamp', sep='\t')