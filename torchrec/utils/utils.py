from torch.nn.utils.rnn import pad_sequence
import torch, yaml, re, importlib, os, logging
import numpy as np
print_logger = logging.getLogger("pytorch_lightning")
print_logger.setLevel(logging.INFO)
def set_color(log, color, highlight=True, keep=False):
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
    model_submodule = ['ae', 'mf', 'seq', 'fm']

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
