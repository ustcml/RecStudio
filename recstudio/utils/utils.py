import hashlib
import importlib
import json
import logging
import os
import random
import re
import time
from collections import OrderedDict
from typing import Optional, Dict, List

import numpy as np
import requests
import yaml
import torch
from tqdm import tqdm
from recstudio.utils.compress_file import extract_compressed_file


LOG_DIR = r"./log/"
DEFAULT_CACHE_DIR = r"./.recstudio/"
URL_UPDATE_URL = r"http://home.ustc.edu.cn/~angus_huang/recstudio/url.yaml"


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.exists(DEFAULT_CACHE_DIR):
    os.makedirs(DEFAULT_CACHE_DIR)


def set_color(log, color, highlight=True, keep=False):
    r"""Set color for log string.

    Args:
        log(str): the 
    """
    if keep:
        return log
    color_set = ['black', 'red', 'green',
                 'yellow', 'blue', 'pink', 'cyan', 'white']
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


def parser_yaml(config_path):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        ret = yaml.load(f.read(), Loader=loader)
    return ret


def color_dict(dict_, keep=True):
    key_color = 'blue'
    val_color = 'yellow'

    def color_kv(k, v, k_f, v_f):
        info = (set_color(k_f, key_color, keep=keep) + '=' +
                set_color(v_f, val_color, keep=keep)) % (k, v)
        return info

    des = 4
    if 'epoch' in dict_:
        start = set_color('Training: ', 'green', keep=keep)
        start += color_kv('Epoch', dict_['epoch'], '%s', '%3d')
    else:
        start = set_color('Testing: ', 'green', keep=keep)
    info = ' '.join([color_kv(k, v, '%s', '%.'+str(des)+'f')
                    for k, v in dict_.items() if k != 'epoch'])
    return start + ' [' + info + ']'


def color_dict_normal(dict_, keep=True):
    dict_ = OrderedDict(sorted(dict_.items()))
    key_color = 'blue'
    val_color = 'yellow'

    def color_kv(k, v, k_f, v_f):
        info = (set_color(k_f, key_color, keep=keep) + '=' +
                set_color(v_f, val_color, keep=keep)) % (k, v)
        return info
    info = '\n'.join([color_kv(k, str(v), '%s', '%s')
                     for k, v in dict_.items()])
    return info


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = ['ae', 'mf', 'seq', 'fm', 'kg']

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['recstudio.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError(
            f'`model_name` [{model_name}] is not the name of an existing model.')
    model_class = getattr(model_module, model_name)
    dir = os.path.dirname(model_module.__file__)
    conf = dict()
    fname = os.path.join(os.path.dirname(dir), 'basemodel', 'basemodel.yaml')
    conf.update(parser_yaml(fname))
    for name in ['all', model_file_name]:
        fname = os.path.join(dir, 'config', name+'.yaml')
        if os.path.isfile(fname):
            conf.update(parser_yaml(fname))
    return model_class, conf


def md5(config: dict):
    s = ''
    for k in sorted(config):
        s += f"{k}:{config[k]}\n"
    md = hashlib.md5(s.encode('utf8')).hexdigest()
    return md


def get_download_url_from_recstore(share_number: str):
    headers = {
        "Host": "recapi.ustc.edu.cn",
        "Content-Type": "application/json",
    }
    data_resource_list = {
        "share_number": share_number,
        "share_resource_number": None,
        "is_rec": "false",
        "share_constraint": {}
    }
    resource = requests.post(
        'https://recapi.ustc.edu.cn/api/v2/share/target/resource/list', 
        json=data_resource_list, headers=headers)
    resource = resource.text.encode("utf-8").decode("utf-8-sig")
    resource = json.loads(resource)
    resource = resource['entity'][0]['number']
    data = {
        "share_number": share_number,
        "share_constraint": {},
        "share_resources_list": [
            resource
        ]
    }
    res = requests.post(
        "https://recapi.ustc.edu.cn/api/v2/share/download", 
        json=data, headers=headers)
    res = res.text.encode("utf-8").decode("utf-8-sig")
    res = json.loads(res)
    download_url = res['entity'][resource] + "&download=download"
    return download_url


def check_valid_dataset(name: str, config: Dict, default_dataset_path=DEFAULT_CACHE_DIR):
    r""" Check existed dataset according to the md5 string.

    Args:
        md5(str): the md5 string of the config.
        default_data_set_path:(str, optional): path of the local cache foler.

    Returns:
        str: download url of the dataset file or the local file path.
    """
    # update url.yaml
    # url_path = os.path.join(DEFAULT_CACHE_DIR, '../url.yaml')
    if not os.path.exists(default_dataset_path):
        os.makedirs(default_dataset_path)

    config_md5 = md5(config)
    cache_file_name = os.path.join(default_dataset_path, "cache", config_md5)
    if os.path.exists(cache_file_name):
        return True, cache_file_name
    else:   # there is no cache file
        # try to find original file
        download_flag = False
        default_dir = os.path.join(default_dataset_path, name)
        for k,v in config.items():
            if k.endswith('feat_name'):
                if not isinstance(v, List) and v is not None:
                    files = [v]
                for f in files:
                    fpath = os.path.join(default_dir, f)
                    if not os.path.exists(fpath):
                        download_flag = True
                        break

        if not download_flag:
            print_logger.info(f"dataset is read from {default_dir}.")
            return False, default_dir
        elif download_flag and (config['url'] is not None):
            if config['url'].startswith('http'):
                print_logger.info(f"will download dataset {name} fron the url {config['url']}.")
                return False, download_dataset(config['url'], name, default_dir)
            else:   # user provide original file
                print_logger.info(f"dataset is read from {config['url']}.")
                return False, config['url']
        elif download_flag and (config['url'] is None):
            raise FileNotFoundError("Sorry, the original dataset file can not be found due to"\
                "there is neither url provided or local file path provided in configuration files"\
                "with the key `url`.")
            


def download_dataset(url: str, name: str, save_dir: str):
    if url.startswith('http'):  # remote
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if "rec.ustc.edu.cn" in url:
                url = get_download_url_from_recstore(share_number=url.split('/')[-1])
                zipped_file_name = f"{name}.zip"
            else:
                zipped_file_name = url.split('/')[-1]
            dataset_file_path = os.path.join(save_dir, zipped_file_name)
            response = requests.get(url, stream=True)
            content_length = int(response.headers.get('content-length', 0))
            with open(dataset_file_path, 'wb') as file, \
                tqdm(desc='Downloading dataset', 
                     total=content_length, unit='iB', 
                     unit_scale=True, unit_divisor=1024) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            extract_compressed_file(dataset_file_path, save_dir)
            os.remove(dataset_file_path)
            return save_dir
        except:
            print("Something went wrong in downloading dataset file.")


def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:
    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.
    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = random.randint(min_seed_value, max_seed_value)
            print_logger.warning(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = random.randint(min_seed_value, max_seed_value)
                print_logger.warning(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        print_logger.warning(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = random.randint(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    print_logger.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def get_dataset_default_config(dataset_name: str) -> Dict:
    dir = os.path.dirname(__file__)
    dataset_config_dir = os.path.join(dir, "../data/config")
    dataset_config_fname = os.path.join(dataset_config_dir, f"{dataset_name}.yaml")
    if os.path.exists(dataset_config_fname):
        config = parser_yaml(dataset_config_fname)
    else:
        print_logger.warning(f"There is no default configuration file for dataset {dataset_name}."\
            "Please make sure that all the configurations are setted in your provided file or the"\
            "configuration dict you've assigned.")
        config = {}
    return config






class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True



FORMAT = '[%(asctime)s] %(levelname)s %(message)s'
# logging.basicConfig(format=FORMAT)
print_logger = logging.getLogger('recstudio')

formatter = logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S")
print_logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
print_logger.addHandler(stream_handler)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, \
    time.strftime("%Y-%m-%d-%H-%M-%S.log", time.localtime())))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
remove_color_filter = RemoveColorFilter()
file_handler.addFilter(RemoveColorFilter())
print_logger.addHandler(file_handler)
