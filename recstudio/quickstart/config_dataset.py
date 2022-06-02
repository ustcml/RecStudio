import os
import pandas as pd


def generate_dataset_config(name: str, dir: str, interaction_file: str, user_id: str, 
    item_id: str, rating:str, timestamp: str, sep='\t', user_file: str=None, item_file: str=None):
    config_file_name = f"{name}.yaml"
    config_path = os.path.join(dir, name)
    config_dict = {
        'url': dir,
        'user_id_field': f"&u {user_id}:token",
        'item_id_field': f"&i {item_id}:token",
        'rating_field': f"&r {rating}:float",
        'time_field': f"&t {timestamp}:float",
        'inter_feat_name': f"{interaction_file}",
        'user_feat_name': f"{user_file}" if user_file else "~",
        'item_feat_name': f"{item_file}" if item_file else "~",

    }
    raise NotImplementedError("Sorry, not supported now, we will implement the function soon.")