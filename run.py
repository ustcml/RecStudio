import os
import time
import argparse
from recstudio.utils import get_model, color_dict_normal, set_color, get_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='RecStudio')
    parser.add_argument('--model', '-m', type=str, default='WRMF', help='model name')
    parser.add_argument('--data_config_path', type=str, help='path of datasets config file')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('--mode', choices=['tune', 'light', 'detail'],
                        default='light', help='flag indiates model tuning')
    args, command_line_args = parser.parse_known_args()
    model_class, model_conf = get_model(args.model)
    parser = model_class.add_model_specific_args(parser)
    args_ = parser.parse_args(command_line_args)
    for k, v in vars(args_).items():
        for arg in command_line_args:
            if k in arg:
                model_conf[k] = v
                break
    log_path = time.strftime(f"{model_class.__name__}-{args.dataset}-%Y-%m-%d-%H-%M-%S.log", time.localtime())
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
