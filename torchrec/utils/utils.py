from torch.nn.utils.rnn import pad_sequence
import torch, yaml, re, importlib, os, logging, urllib,zipfile
import numpy as np
print_logger = logging.getLogger("pytorch_lightning")
print_logger.setLevel(logging.INFO)
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

def check_valid_dataset(user_data_dir, default_data_dir, dataset_name: str, atom_file_list: list):
    # 数据集文件夹分为用户提供的和系统内置的. 
    # 当用户提供文件夹时,优先处理用户提供的文件夹内的原子文件. 
    # 如果用户提供的文件夹内没有文件或者用户没有提供文件夹路径,那么则检查内置的文件夹内有没有对应数据集的文件夹
    # 1.判断是否存在文件夹,若不存在文件夹,那么下载对应cache文件
    # 2.若存在文件夹,检查是否存在 recstudio_{dataset}.cache 缓存文件.若不存在: 
        # 1. 检查文件夹内是否存在所有的原子文件,如果不存在,重新下载原子文件.而后进行处理成缓存文件并存储;
        # 2. 若文件夹内存在所有原子文件,则直接处理成缓存文件并存储.
    default_dataset_dir = os.path.join(default_data_dir, dataset_name)
    download_url = check_remote_server(dataset_name)
    if (user_data_dir is not None) and (os.path.exists(os.path.join(user_data_dir, dataset_name))):
        user_dataset_dir = os.path.join(user_data_dir, dataset_name)
        if os.path.exists(user_dataset_dir):
            for f in atom_file_list:
                if not os.path.exists(os.path.join(user_dataset_dir, f)):
                    if download_url is not None:
                        download_dataset(download_url, default_dataset_dir)
                        return "download_cache"
                    else: 
                        raise FileNotFoundError('File not found.')
            return "user_defined" #用户提供的原子文件完备,直接处理原子文件
    else:
        if not os.path.exists(os.path.join(default_dataset_dir, "recstudio_{}.cache".format(dataset_name))):
            if download_url is not None:
                download_dataset(download_url, default_dataset_dir)
                return "download_cache"
            else:
                raise FileNotFoundError("Dataset file not found.")
        else:
                return "cache_found"

def check_remote_server(dataset_name: str) -> str:
    remote_dataset_link = {
        "ml-100k": "link1",
        "ml-1m": "link2",
        "ml-10m": "link3",
        "ml-20m": "link4",
        "gowalla": "link5",
        "amazon-books": "link6",
        "amazon-electronics": "link7",
        "yelp": "link8"
    }
    if dataset_name in remote_dataset_link:
        return remote_dataset_link[dataset_name]
    else:
        return None

def download_dataset(url:str, save_dir: str):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dataset_file_path = os.path.join(save_dir, "_temp_cache.zip")
        urllib.urlretrieve(url, dataset_file_path)
        zip_file = zipfile.ZipFile(dataset_file_path)
        zipfile.extractall(save_dir)
        zip_file.close()
        os.remove(dataset_file_path)
    except:
        print("Something went wrong in downloading dataset file.")
    
    
def test():
    user_data_dir = "testdataset"
    default_data_dir = "dataset" 
    dataset_name = "ml-100k"
    atom_file_list = ["ml-100k.inter"]