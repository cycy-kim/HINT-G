import os
import time

import torch

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
    # xaifg는 cpu가 더 빠를때도 많음

    # node
    args.dataset = 'syn3'
    
    # graph
    # args.dataset = 'BA-2motif'
    # args.hiddens = '25-25-25'
    # args.concat = False
    # args.bn = False

    args.gnn_type = 'supervised' # 'supervised' or 'unsupervised'
    args.hidden_dim = 128 # unsupervised일때만
    args.task = 'neg' # 'pos' or 'neg'
    args.setting = 1 # neg일때 setting 잘보고하기
    
    # hyper params grid
    iters = [2, 4]
    scales = [10, 100, 1000, 10000, 100000, 1000000]
    #######################


    args.gnn_task = 'node' if args.dataset[:3] == 'syn' else 'graph'
    grid_result_file_name = f"{args.dataset}_{args.gnn_type}_{args.task}_grid_result.txt"


    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type

    explainer = XAIFG_Node(args=args)

    if not os.path.exists(grid_result_file_name):
        with open(grid_result_file_name, "w") as f:
            f.write(f'nodes: {explainer.data.nodes}\n')
            f.write("iter, scale, rocauc\n")  # 헤더 추가 (필요하면)

    with open(grid_result_file_name, "w") as f:
        for iter in iters:
            for scale in scales:        
                args.iter = iter
                args.scale = scale
                
                infer_start_time = time.time()
                rocauc, acc, precision, recall = explainer.edge_influences()
                infer_end_time = time.time()

                infer_time = infer_end_time - infer_start_time
                # print(f'Inference time: {infer_time:.2f} seconds')
                f.write(f'iter {iter}, scale {scale}, rocauc {rocauc}, time {infer_time:.2f}s\n')
                f.flush()

