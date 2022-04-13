import os, nni, argparse
from recstudio.utils.utils import get_model, print_logger, color_dict_normal, parser_yaml
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='WRMF', help='model name')
    parser.add_argument('--data_dir', type=str, default='datasets', help='directory of datasets')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='dataset name')
    parser.add_argument('--mode', choices=['tune', 'light', 'detail'], default='light', help='flag indiates model tuning')
    args, commond_line_args = parser.parse_known_args()
    model_class, model_conf = get_model(args.model)
    parser = model_class.add_model_specific_args(parser)
    args_ = parser.parse_args(commond_line_args)
    for opt in commond_line_args:
        key = opt.split('=')[0].strip('-')
        value = getattr(args_, key)
        model_conf[key] = value
    if args.mode == 'tune':
        tune_para = nni.get_next_parameter()
        for k, v in tune_para.items():
            if '/' in k:
                model_conf[k.split('/')[-2]] = v
            else:
                model_conf[k] = v
    model = model_class(model_conf)
    print_logger.info(color_dict_normal(model_conf, args.mode =='tune'))
    dataset_filename = f"{args.data_dir}/{args.dataset}/{args.dataset}.yaml"
    if not os.path.isfile(dataset_filename):
        raise ValueError('Please provide dataset description in a yaml file')
    datasets = model.load_dataset(dataset_filename)
    model.fit(*datasets[:2], run_mode=args.mode)
    model.evaluate(datasets[-1])
    

