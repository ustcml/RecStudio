import os, nni
import os
import recstudio
from recstudio.utils import get_model, print_logger, color_dict_normal, parser_yaml

DATASET_DEMO_PATH = os.path.join(os.path.dirname(__file__),'../dataset_demo')
def run(model:str, data_dir:str=DATASET_DEMO_PATH, dataset:str='ml-100k', mode:str='light', config_file:str=None):
    model_class, model_conf = get_model(model)
    if config_file is not None:
        user_config = parser_yaml(config_file)
        model_conf.update(user_config)
    if mode == 'tune':
        tune_para = nni.get_next_parameter()
        for k, v in tune_para.items():
            if '/' in k:
                model_conf[k.split('/')[-2]] = v
            else:
                model_conf[k] = v
    model = model_class(model_conf)
    print_logger.info(color_dict_normal(model_conf, mode=='tune'))
    dataset_filename = f"{data_dir}/{dataset}/{dataset}.yaml"
    if not os.path.isfile(dataset_filename):
        raise ValueError('Please provide dataset description in a yaml file')
    datasets = model.load_dataset(dataset_filename)
    model.fit(*datasets[:2], run_mode=mode)
    model.evaluate(datasets[-1])
    

