import argparse
import numpy as np
import os
import random
import torch

def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument("--dataset", type=str, default='syn', help="Dataset string, syn1~4, BA-2motif")
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--seed',type=int, default=42, help='seed')
    parser.add_argument('--setting', type=int, default=1)  # 1 is used for GNNExplainer and PGExplainer paper.

    # parameters for GCN
    parser.add_argument('--order', type=str, default='AW')  #
    parser.add_argument('--embnormalize', type=bool, default=True)  #
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--initializer', default='glorot')
    
    parser.add_argument('--early_stop', type=int, default= 2000, help='early_stop')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--dropout',type=float, default=0.0, help='dropout rate (1 - keep probability).')
    parser.add_argument('--hiddens', type=str, default='20-20-20') # for gcn
    parser.add_argument("--lr", type=float, default=0.001,help='initial learning rate.')
    parser.add_argument('--bn', type=bool, default=True)
    parser.add_argument('--concat', type=bool, default=True)
    parser.add_argument('--valid', type=bool, default=False)
    parser.add_argument('--sample_bias',type=float, default=0.0, help='bias for sampling from 0-1')
    parser.add_argument('--budget', type=float, default=-1.0, help='coefficient for size constriant')
    parser.add_argument('--weight_decay',type=float, default=0.0, help='Weight for L2 loss on embedding matrix.')

    parser.add_argument('--save_model',type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='checkpoints/gcn')


    parser.add_argument('--topk', type=lambda v: int(v) if v.isdigit() else float(v), default=1)
    parser.add_argument('--train_ratio', type=float, default=0.1)

    args, _ = parser.parse_known_args()

    return args

args = get_params()
params = vars(args)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

seed = args.seed
if seed>0:
    random.seed(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

dtype = torch.float32
if args.dtype == 'float64':
    dtype = torch.float64

eps = 1e-7
