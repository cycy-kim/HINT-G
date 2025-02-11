import torch
import numpy as np
import random

from models import CFGNNExplainer
from dataset import SyntheticDataset
from utils import args


def get_graph(args, data:SyntheticDataset, nodeid, rand_remove=False):
    """
    Args:
        args (_type_): _description_
        data (): _description_
        nodeid (int): _description_
    
    Returns:
        graph (tuple): feature, adj, label
    """
    if args.gnn_task == 'node':
        remap = data.remap
        remapped_nodeid = remap[nodeid]
        adj = data.sub_support_tensors[remapped_nodeid].to_dense()
        if rand_remove:
            original_sub_support = data.sub_support_tensors[remapped_nodeid]      
            indices = original_sub_support.coalesce().indices()
            values = original_sub_support.coalesce().values()
            size = original_sub_support.coalesce().size()

            deletion_candidates = []
            for edge_index in range(len(values)):
                src, des = indices[0][edge_index].item(), indices[1][edge_index].item()
                if src >= des:
                    continue
                if data.sub_edge_labels[remap[nodeid]].todense()[src,des] or data.sub_edge_labels[remap[nodeid]].todense()[des,src]:
                    deletion_candidates.append((src, des))

            
            edges_to_delete = random.sample(deletion_candidates, 1)
            
            src_del, des_del = edges_to_delete[0]
            mask = ~(((indices[0] == src_del) & (indices[1] == des_del)) | ((indices[0] == des_del) & (indices[1] == src_del)))
            new_indices = indices[:, mask]
            new_values = values[mask]
            del_sub_support = torch.sparse_coo_tensor(new_indices, new_values, size)
            adj = del_sub_support.to_dense()
        features = data.sub_features[remapped_nodeid]
        labels = np.argmax(data.sub_labels[remapped_nodeid], axis=-1)
        graph = (features, adj, labels)

    elif args.gnn_task == 'graph':
        adj = torch.tensor(data.adjs[nodeid], dtype=torch.float32, device=args.device)
        features = torch.tensor(data.feas[nodeid], dtype=torch.float32, device=args.device)
        labels = torch.tensor(data.labels[nodeid], dtype=torch.float32, device=args.device)
        graph = (features, adj, labels)

    return graph


if __name__=='__main__':
    """"""
    args.device = 'cuda'
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.task = 'pos'           # 'pos' or 'neg'
    args.gnn_type = 'supervised' # 'supervised' or 'unsupervised'
    
    
    args.epochs = 200
    args.lr = 0.01
    args.cf_optimizer = "SGD"
    args.beta = 0.5
    args.n_momentum = 0.0
    # ------------- #

    # graph classification #
    # args.dataset = 'BA-2motif'
    # args.hiddens = '20-20-20'
    # args.bn = False     # model
    # args.concat = False   # model
    # args.setting = 1
    # args.model_weight_path = args.save_path + args.dataset # +'_'+ args.gnn_type

    
    # node classification #
    args.dataset = 'syn3'
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type
    args.setting = 1 # default to 1, syn3 neg


    if args.dataset[:3] == 'syn':
        args.gnn_task = 'node'
    else:
        args.gnn_task = 'graph'

    data = SyntheticDataset(args=args)
    data.cuda()

    if args.gnn_task =='node':
        data.prepare_inductive()
        data.cuda() # subgraphs
        from models import GCN_Node as GCN
        num_classes = data.y_train.shape[1]
        model = GCN(args=args, input_dim=data.features.shape[1], output_dim=num_classes).to(args.device)
        model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))

    elif args.gnn_task =='graph':
        from models import GCN_Graph as GCN
        num_classes = data.labels.shape[1]
        model = GCN(args=args, input_dim=data.feas.shape[-1], output_dim=num_classes).to(args.device)
        model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))

    allnodes = data.nodes
    target_nodeid = 0
    rocaucs = []
    for nodeid in allnodes:
        if args.dataset == 'syn4' and args.task=='neg':
            graph = get_graph(args, data, nodeid, rand_remove=True)
        else:
            graph = get_graph(args, data, nodeid)
        
        _, adj, label = graph
        if args.task == 'pos' and label[target_nodeid].item() == 0: # pos인데 motif 없는거
            continue
        if args.task == 'neg' and label[target_nodeid].item() == 1 and args.dataset != 'syn4': # neg인데 motif 있는거
            continue
        if args.task == 'neg' and label[target_nodeid].item() == 0 and args.dataset == 'syn4':
            continue

        explainer = CFGNNExplainer(args, model, graph, num_classes).to(args.device)
        
        # new_idx = target_nodeid = 0?
        # adj_recons, adj_score = explainer.explain(args.cf_optimizer, nodeid, 0, args.lr, args.n_momentum, args.epochs)
        rocauc = explainer.get_explanation_score(nodeid, data, target_nodeid=0)
        if rocauc != -1:
            rocaucs.append(rocauc)
        print(f'nodeid {nodeid}, rocauc: {rocauc}')
    print(f'mean roc auc : {np.mean(rocaucs)}')
        