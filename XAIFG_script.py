import torch

import time

from models import XAIFG
from utils import args


"""
syn3:       20, 1000000
syn4:       3,  100000
BA-2motif:  10, 1000
"""



if __name__== '__main__':

    ###### edit here ######
    args.device = 'cuda'

    ###
    # node
    args.dataset = 'syn3'
    
    # graph
    # args.dataset = 'BA-2motif'
    # args.hiddens = '25-25-25'
    # args.concat = False
    # args.bn = False
    ###


    ###
    args.gnn_type = 'unsupervised' # 'supervised' or 'unsupervised'
    args.hidden_dim = 128 # only used when unsupervised
    args.task = 'pos' # 'pos' or 'neg'
    args.setting = 1 # 1로 고정 / pos neg에 맞게 dataset/dataset.py에 NODE_SETTINGS 수정
    ###

    args.iter = 10
    args.scale = 1000

    #######################


    args.gnn_task = 'node' if args.dataset[:3] == 'syn' else 'graph'
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type

    explainer = XAIFG(args=args)
    infer_start_time = time.time()
    rocauc, acc, precision, recall = explainer.edge_influences()
    infer_end_time = time.time()
    print(rocauc)

    infer_time = infer_end_time - infer_start_time
    print(f'Inference time: {infer_time:.2f} seconds')
