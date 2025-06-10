import torch

import time

from models import XAIFG_Node
from utils import args

"""
syn3:       20, 1000000
syn4:       3,  100000
BA-2motif:  10, 1000
"""

if __name__== '__main__':

    ###### edit here ######
    args.device = 'cuda'
    

    # node
    args.dataset = 'syn3'
    
    # graph
    # args.dataset = 'BA-2motif'
    # args.hiddens = '25-25-25'
    # args.concat = False
    # args.bn = False

    args.gnn_type = 'supervised' # 'supervised' or 'unsupervised'
    args.hidden_dim = 128 # only used in unsupervised
    args.task = 'neg' # 'pos' or 'neg'
    args.setting = 1 # neg일때 setting 잘보고하기
    
    # hyper params
    args.iter = 20
    args.scale = 100000

    #######################


    args.gnn_task = 'node' if args.dataset[:3] == 'syn' else 'graph'

    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type

    explainer = XAIFG_Node(args=args)
    infer_start_time = time.time()
    rocauc, acc, precision, recall = explainer.edge_influences()
    infer_end_time = time.time()
    print(rocauc)

    infer_time = infer_end_time - infer_start_time
    print(f'Inference time: {infer_time:.2f} seconds')
