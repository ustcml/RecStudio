from typing import Dict, Union
from recstudio.utils import get_model, print_logger, color_dict_normal, set_color, parser_yaml


def run(model: str, dataset: str, model_config: Union[Dict, str]=None, data_config: Union[Dict, str]=None, **kwargs):
    model_class, model_conf = get_model(model)
    if model_config is not None:
        if isinstance(model_config, Dict):
            model_conf.update(model_config)
        elif isinstance(model_config, str):
            model_conf.update(parser_yaml(model_config))
        else:
            raise TypeError("expecting `config` to be Dict or string," \
                        f"while get {type(model_config)} instead.")

    if kwargs is not None:
        model_conf.update(kwargs)

    model = model_class(model_conf)
    dataset_class = model._get_dataset_class()
    datasets = dataset_class(name=dataset, config=data_config).build(**model_conf)
    print_logger.info(f"{datasets[0]}")
    print_logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
    model.fit(*datasets[:2], run_mode='light')
    model.evaluate(datasets[-1])
