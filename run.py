import os, argparse
from recstudio.utils import get_model, print_logger, color_dict_normal, set_color
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='WRMF', help='model name')
    parser.add_argument('--data_dir', type=str, default='datasets', help='directory of datasets')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('--mode', choices=['tune', 'light', 'detail'], default='light', help='flag indiates model tuning')
    args, commond_line_args = parser.parse_known_args()
    model_class, model_conf = get_model(args.model)
    # parser = model_class.add_model_specific_args(parser)
    args_ = parser.parse_args(commond_line_args)
    for opt in commond_line_args:
        key = opt.split('=')[0].strip('-')
        value = getattr(args_, key)
        model_conf[key] = value
    # if args.mode == 'tune':
    #     tune_para = nni.get_next_parameter()
    #     for k, v in tune_para.items():
    #         if '/' in k:
    #             model_conf[k.split('/')[-2]] = v
    #         else:
    #             model_conf[k] = v
    model = model_class(model_conf)
    # dataset_filename = os.path.join(args.data_dir, args.dataset, f"{args.dataset}.yaml")
    # if not os.path.isfile(dataset_filename):
    #     raise ValueError('Please provide dataset description in a yaml file')

    dataset_class = model._get_dataset_class()
    datasets = dataset_class(name=args.dataset).build(**model_conf)
    # print_logger.info(f"\n{set_color('Dataset Info','green')}: \n{datasets[0]}")
    print_logger.info(f"{datasets[0]}")
    # print_logger.info(datasets[0])
    # datasets = model.load_dataset(dataset_filename)

    print_logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, args.mode =='tune'))
    model.fit(*datasets[:2], run_mode=args.mode)
    model.evaluate(datasets[-1])
    


    # model1 = BPR()
    # model2 = FM(model1)
    # # model2.config_sampler(model1)
    # config["sampler"] = ['uniform', 'popularity', 'retriever_dns', 'retriever_importancesampling']
    # config["retriver"] = ['retriver', 'uniform'm None]
    # model2.sampler = UniformSampler()

    # model.retriever = [retriever, uniform_retriever, None] 
    
    # model2.fit()

# TransformEncoder = Sequential()
# TransformEncoder.get_query_feat = lambda batch: List    # 

# class MultiInputSequential():
#     def __init__(self, feat):
#         self.encoder_dict = ModuleDict({
#             'key': encoder,
#         })

#     def forward(self, dic):
#         dict_data = {}
#         for k in dic:
#             dict_data[k] = self.encoder_dict[k](dic[k])

#         self.pool(dict_data)

#     def pool(self, dict_data):
#         pass

    
# class QueryEncoder():
#     def __init__(self, ):

#     def _get_query_feat(self):

#     def forward(self, batch):
#         data = self._input(batch)
# 
  