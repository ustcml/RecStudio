import os
import torch
import time
import torch
import argparse
from recstudio.utils import get_model, color_dict_normal, set_color, get_logger
import nni


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='RecStudio')
    parser.add_argument('--model', '-m', type=str, default='WRMF', help='model name')
    parser.add_argument('--data_config_path', type=str, help='path of datasets config file')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('--mode', choices=['tune', 'light', 'detail'],
                        default='light', help='flag indiates model tuning')
    args, command_line_args = parser.parse_known_args()
    model_class, model_conf = get_model(args.model)#加载各个yaml文件参数
    parser = model_class.add_model_specific_args(parser)
    args_ = parser.parse_args(command_line_args)#add_model_specific_args里的参数以及commmand_line_args里的参数
    print(command_line_args)
    for k, v in vars(args_).items():#这里会用命令行额外指定的参数覆盖掉yaml文件里的参数
        for arg in command_line_args:
            if k in arg:
                model_conf[k] = v
                break

    if args.mode == 'tune':
        tune_para = nni.get_next_parameter()
        for k, v in tune_para.items():
            if '/' in k:
                model_conf[k.split('/')[-2]] = v
            else:
                model_conf[k] = v
    #model_conf里的参数是最终用来训练的各种参数
    log_path = time.strftime(f"{model_class.__name__}-{args.dataset}-%Y-%m-%d-%H-%M-%S.log", time.localtime())

    if args.mode == 'tune':
        from recstudio import LOG_DIR
        if not os.path.exists(LOG_DIR + nni.get_experiment_id() + "/"):
            os.makedirs(LOG_DIR + nni.get_experiment_id() + "/")
        log_path = nni.get_experiment_id() + "/" + nni.get_trial_id() + "-" + log_path
        model_conf['save_path'] = model_conf['save_path'] + nni.get_experiment_id() + "/"
    logger = get_logger(log_path)

    model = model_class(model_conf)

    dataset_class = model_class._get_dataset_class()

    if args.data_config_path is not None:
        dataset_name = os.path.basename(args.data_config_path)
        if args.dataset is not None:
            logger.warning(f"Dataset will be named as {dataset_name} (same as config file name).")
    else:
        dataset_name = args.dataset
    datasets = dataset_class(name=dataset_name, config=args.data_config_path).build(**model_conf)

    logger.info(f"{datasets[0]}")

    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, args.mode == 'tune'))
    model.fit(*datasets[:2], run_mode=args.mode)
    model.evaluate(datasets[-1])
