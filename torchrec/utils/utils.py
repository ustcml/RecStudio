from torch.nn.utils.rnn import pad_sequence
import torch, yaml, re, importlib, os, logging, urllib, zipfile, hashlib, requests
import numpy as np
from tqdm import tqdm

print_logger = logging.getLogger("pytorch_lightning")
print_logger.setLevel(logging.INFO)
DEFAULT_CACHE_DIR = r"./.recstudio/dataset/"

def set_color(log, color, highlight=True, keep=False):
    r"""Set color for log string.
    
    Args:
        log(str): the 
    """
    if keep:
        return log
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
        info = (set_color(k_f, key_color, keep=keep) +'=' + set_color(v_f, val_color, keep=keep)) % (k, v)
        return info

    des = 4
    if 'epoch' in dict_:
        start = set_color('Training: ', 'green', keep=keep)
        start += color_kv('Epoch', dict_['epoch'], '%s', '%3d')
    else:
        start = set_color('Testing: ', 'green', keep=keep)
    info = ' '.join([ color_kv(k, v, '%s', '%.'+str(des)+'f') for k, v in dict_.items() if k != 'epoch'])   
    return start + ' ['+ info + ']'

def color_dict_normal(dict_, keep=True):
    key_color = 'blue'
    val_color = 'yellow'
    def color_kv(k, v, k_f, v_f):
        info = (set_color(k_f, key_color, keep=keep) +'=' + set_color(v_f, val_color, keep=keep)) % (k, v)
        return info
    info = '\n'.join([ color_kv(k, str(v), '%s', '%s') for k, v in dict_.items()])   
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
        module_path = '.'.join(['torchrec.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    dir = os.path.dirname(model_module.__file__)
    conf = dict()
    fname = os.path.join(os.path.dirname(dir), 'basemodel.yaml')
    conf.update(parser_yaml(fname))
    for name in ['all', model_file_name]:
        fname = os.path.join(dir, 'config', name+'.yaml')
        if os.path.isfile(fname):
            conf.update(parser_yaml(fname))
    return model_class, conf
 
def md5(file_path: str):
    r"""Generate md5 for a file."""
    with open(file_path, 'rb') as fp:
        data = fp.read()
    file_md5= hashlib.md5(data).hexdigest()
    return file_md5

def check_valid_dataset(md5: str, default_dataset_path=DEFAULT_CACHE_DIR):
    md2links = parser_yaml(os.path.join(os.path.dirname(__file__), 'url.yaml'))
    if not os.path.exists(default_dataset_path):
        os.makedirs(default_dataset_path)
    local_files = os.listdir(default_dataset_path)
    for fname in local_files:   # cache found in local
        if md5 == fname.split('.')[0].split('_')[2]:
            return os.path.join(default_dataset_path, fname)
    if md5 in md2links: # cache in remote.
        return md2links[md5]
    else:
        return None

def download_dataset(url:str, md:str, save_dir: str=DEFAULT_CACHE_DIR):
    if url.startswith('https://'):  # remote
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dataset_file_path = os.path.join(save_dir, "_temp_cache.zip")
            response = requests.get(url,stream=True)
            content_length = int(response.headers.get('content-length', 0))
            with open(dataset_file_path, 'wb') as file, tqdm(desc='Downloading dataset', total=content_length, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            zip_file = zipfile.ZipFile(dataset_file_path)
            zip_file.extractall(save_dir)
            zip_file.close()
            os.remove(dataset_file_path)
            return os.path.join(save_dir, "recstudio_dataset_{}.cache".format(md))
        except:
            print("Something went wrong in downloading dataset file.")
    else:   # local
        return url

    
    
def test():
    user_data_dir = "testdataset"
    default_data_dir = "dataset" 
    dataset_name = "ml-100k"
    atom_file_list = ["ml-100k.inter"]