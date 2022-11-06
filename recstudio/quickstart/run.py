import time
from typing import Dict, Union
from recstudio.utils import get_model, color_dict_normal, set_color, parser_yaml, get_logger


def run(model: str, dataset: str, model_config: Union[Dict, str]=None, data_config: Union[Dict, str]=None, **kwargs):
    model_class, model_conf = get_model(model)
    if model_config is not None:
        if isinstance(model_config, Dict):
            model_conf.update(model_config)
        elif isinstance(model_config, str):
            model_conf.update(parser_yaml(model_config))
        else:
            raise TypeError(f"expecting `config` to be Dict or string, while get {type(model_config)} instead.")

    if kwargs is not None:
        model_conf.update(kwargs)

    log_path = time.strftime(f"{model_class.__name__}-{dataset}-%Y-%m-%d-%H-%M-%S.log", time.localtime())
    logger = get_logger(log_path)
    model = model_class(model_conf)
    dataset_class = model_class._get_dataset_class()
    datasets = dataset_class(name=dataset, config=data_config).build(**model_conf)
    logger.info(f"{datasets[0]}")
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
    model.fit(*datasets[:2], run_mode='light')
    model.evaluate(datasets[-1])
