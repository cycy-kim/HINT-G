import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np
import itertools
import random

from models import XAIFG
from models import GCN_Graph, GCN_Node
from dataset import SyntheticDataset
from utils import args


def grid_del_pred_change():
    """
    Tree-grid에서 실험대로 edge아무거나 1개 지웠을때 실제 출력이 바뀌는지 아닌지   
    랜덤하게 여러개 지워보고.. 1개로 안되면 여러개도 지워보고 등등 ㄱㄱ
    """
    args.device = 'cuda'
    args.dataset = 'syn4'
    args.gnn_type = 'supervised' 
    args.task = 'neg'
    args.setting = 1
    
    args.gnn_task = 'node' if args.dataset[:3] == 'syn' else 'graph'
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type

    data = SyntheticDataset(args=args)
    data.prepare_inductive()
    data.cuda()

    explainer = XAIFG(args=args)
    nodes = data.nodes

    num_changed = 0
    num_not_changed = 0
    for nodeid in nodes:
        original_feature_tensor = data.sub_features[data.remap[nodeid]]
        original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
        # label = torch.tensor(np.argmax(sub_labels[nodeid]))
        # label = torch.tensor(data.sub_labels[data.remap[nodeid]])
        indices = original_support_tensor.coalesce().indices()
        values = original_support_tensor.coalesce().values()
        size = original_support_tensor.coalesce().size()
        

        deletion_candidates = []
        for edge_index in range(len(values)):
            src, des = indices[0][edge_index].item(), indices[1][edge_index].item()
            if src >= des:
                continue
            if data.sub_edge_labels[data.remap[nodeid]].todense()[src,des] or data.sub_edge_labels[data.remap[nodeid]].todense()[des,src]:
                deletion_candidates.append((src, des))
        
        
        edges_to_delete = random.sample(deletion_candidates, k=1)
        # edges_to_delete = random.sample(deletion_candidates, 1) # 일단 k=1 고정

        if edges_to_delete:
            mask = torch.ones(len(values), dtype=torch.bool, device=args.device)
            for src_del, des_del in edges_to_delete:
                mask &= ~(((indices[0] == src_del) & (indices[1] == des_del)) | ((indices[0] == des_del) & (indices[1] == src_del)))

            new_indices = indices[:, mask]
            new_values = values[mask]
            del_original_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)

            original_output = explainer.model((original_feature_tensor, original_support_tensor), training=False)
            perturbed_output = explainer.model((original_feature_tensor, del_original_support_tensor), training=False)

            original_pred = torch.argmax(original_output[0], dim=0).item()
            target_pred = torch.argmax(perturbed_output[0], dim=0).item()

            if original_pred == target_pred:
                print(f'nodeid {nodeid}, prediction did not change')
                num_not_changed += 1
            else:
                print(f'nodeid {nodeid}, prediction changed')
                num_changed += 1

    print('-----------')
    print(f'num of graph with pred changed: {num_changed}')
    print(f'not changed: {num_not_changed}')

def grid_del_pred_change_mkII():
    """
    motif 12개중에 한개씩 지워보고 12번중에 몇번 바뀌는지
    """
    args.device = 'cuda'
    args.dataset = 'syn4'
    args.gnn_type = 'supervised' 
    args.task = 'neg'
    args.setting = 1
    
    args.gnn_task = 'node' if args.dataset[:3] == 'syn' else 'graph'
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type

    data = SyntheticDataset(args=args)
    data.prepare_inductive()
    data.cuda()

    explainer = XAIFG(args=args)
    nodes = data.nodes


    changed_ratio_sum = 0.0
    num_changed = 0
    for nodeid in nodes:
        original_feature_tensor = data.sub_features[data.remap[nodeid]]
        original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
        # label = torch.tensor(np.argmax(sub_labels[nodeid]))
        # label = torch.tensor(data.sub_labels[data.remap[nodeid]])
        indices = original_support_tensor.coalesce().indices()
        values = original_support_tensor.coalesce().values()
        size = original_support_tensor.coalesce().size()
        

        deletion_candidates = []
        for edge_index in range(len(values)):
            src, des = indices[0][edge_index].item(), indices[1][edge_index].item()
            if src >= des:
                continue
            if data.sub_edge_labels[data.remap[nodeid]].todense()[src,des] or data.sub_edge_labels[data.remap[nodeid]].todense()[des,src]:
                deletion_candidates.append((src, des))
        
        num_changed = 0
        for src_del, des_del in deletion_candidates:
            mask = ~(((indices[0] == src_del) & (indices[1] == des_del)) | ((indices[0] == des_del) & (indices[1] == src_del)))
            new_indices = indices[:, mask]
            new_values = values[mask]
                
            del_original_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)

            original_output = explainer.model((original_feature_tensor, original_support_tensor), training=False)
            perturbed_output = explainer.model((original_feature_tensor, del_original_support_tensor), training=False)

            original_pred = torch.argmax(original_output[0], dim=0).item()
            target_pred = torch.argmax(perturbed_output[0], dim=0).item()

            if original_pred != target_pred:
                num_changed += 1
                # print(f'nodeid {nodeid}, prediction did not change')
            # else:
                # print(f'nodeid {nodeid}, prediction changed')
                
        change_ratio = num_changed / len(deletion_candidates)
        changed_ratio_sum += change_ratio
    print('-----------')
    print(f'mean ratio of pred changed: {changed_ratio_sum / len(nodes)}')
    # print(f'num of graph with pred changed: {num_changed}')
    # print(f'not changed: {num_not_changed}')


if __name__=="__main__":
    # grid_del_pred_change()
    grid_del_pred_change_mkII()
