from typing import *
from argparse import ArgumentParser, SUPPRESS, ArgumentDefaultsHelpFormatter

from recstudio.utils.utils import parser_yaml, get_model

_CHOICES = {
    'accelerator': ['cpu', 'gpu', 'dp', 'ddp'],
    'early_stop_mode': ['min', 'max'],
    'init_method': ['xavier_normal', 'normal', 'xavier_uniform'],
    'sampling_method': ['none', 'sir', 'dns', 'toprand', 'top&rand', 'brute'],
    'scheduler': ['onplateau', 'exponential'],
    'learner': ['adam', 'adagrad', 'sgd', 'rmsprop', 'sparse_adam'],
    'sampling_sampler': ['uniform', 'pop', 'midx-uni', 'midx-pop', 'cluster-uni', 'cluster-pop']
}

_ARG_MAP = {}


def value2type(key, value):
    arg_args = {'type': type(value),}
    if value is None:
        arg_args['type'] = str
    elif isinstance(value, list):
        arg_args['type'] = type(value[0])
        arg_args['nargs'] = "*"
        arg_args['help'] = f"List({type(value[0]).__name__}) type, e.g. '--{key} v1 v2 v3'"
    elif isinstance(value, bool):
        arg_args['type'] = int
        arg_args['choices'] = [0,1]
        arg_args['help'] = "Bool type, 1-True, 0-False"
    else:
        pass
    if (not 'choices' in arg_args) and (key in _CHOICES):
        arg_args['choices'] = _CHOICES[key]
    return arg_args



def dict2arguments(config: Dict, parser: ArgumentParser) -> ArgumentParser:
    groups = list(config.keys())
    global _ARG_MAP
    args = []
    for g in groups:
        group_p = parser.add_argument_group(g)
        for k, v in config[g].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if not f"{k}_{k2}" in args:
                        group_p.add_argument(f"--{k}_{k2}", dest=f"{g}/{k}/{k2}", **value2type(f"{k}_{k2}", v2))
                        args.append(f"{k}_{k2}")
                    else:   # argument name conflicts
                        group_p.add_argument(f"--{g}_{k}_{k2}", dest=f"{g}/{k}/{k2}", **value2type(f"{g}_{k}_{k2}", v2))
                        args.append(f"{g}_{k}_{k2}")
            else:
                if k not in args:
                    group_p.add_argument(f"--{k}", dest=f"{g}/{k}", **value2type(k, v))
                    args.append(k)
                else:   # argument name conflicts
                    group_p.add_argument(f"--{g}_{k}", dest=f"{g}/{k}", **value2type(f"{g}_{k}", v))
                    args.append(f"{g}_{k}")
    return parser


def get_default_parser() -> ArgumentParser:
    # _dir = os.path.dirname(os.path.realpath(__file__))
    # base_conf_file = os.path.join(_dir, "../model/basemodel/basemodel.yaml")
    # base_config = parser_yaml(base_conf_file)
    parser = ArgumentParser(
        prog='RecStudio',
        description="RecStudio Argparser",
        argument_default=SUPPRESS,
        formatter_class=ArgumentDefaultsHelpFormatter)
    group = parser.add_argument_group('main')
    group.add_argument('--model', '-m', type=str, default='BPR', help='model name')
    group.add_argument('--dataset', '-d', type=str, default='ml-100k', help='dataset name')
    group.add_argument('--data_config_path', type=str, default=None, help='path of datasets config file')
    group.add_argument('--mode', choices=['tune', 'light', 'detail'],
                        default='light', help='flag indiates model tuning')
    return parser


def add_model_arguments(parser: ArgumentParser, model_name: str) -> ArgumentParser:
    r""" Add arguments to parser for specific model name.

    Args:
        parser(ArgumentParser): the parser to be added with arguments
        model_name(str): specific model name

    Returns:
        ArgumentParser: parser with additional arguments
    """
    model_class, model_conf = get_model(model_name)
    parser = dict2arguments(model_conf, parser)
    return parser


def parser2nested_dict(parser: ArgumentParser, command_line_args: List[str],) -> Dict:
    r""" Convert the namespace to nested dict """
    args = parser.parse_args(command_line_args)
    args = vars(args)
    config = dict()
    for k, v in args.items():
        if '/' in k:
            sub_k = k.split('/')
            config_ = config
            for i, _k in enumerate(sub_k):
                if _k not in config_:
                    if i < (len(sub_k) - 1):
                        config_.update({_k: dict()})
                    else:
                        config_.update({_k: v})
                config_ = config_[_k]
        else:   # exclude the arguments in `main` group
            pass
    return config


if __name__ == "__main__":
    parser = get_default_parser()
    # res = parser.parse_args(["--model", "BPR", "--batch_size", "512"])

    conf = parser2nested_dict(parser, ["--model", "BPR", "--batch_size", "512", "--sampling_sampler", "pop"])
    print(conf)