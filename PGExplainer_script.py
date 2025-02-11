import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np
import time

from models import PGExplainer_Graph, PGExplainer_Node
from models import GCN_Graph, GCN_Node
from dataset import SyntheticDataset
from utils import args

"""
ba2motif는 다 됐음

node는 embed 그거 마저 해야됨
"""
@torch.no_grad()
def test_explainer(explainer, data:SyntheticDataset):
    rocauc = explainer.get_explanation_scores(data)
    return rocauc

def train_explainer(args, explainer, data:SyntheticDataset):
    """"""
    explainer.train()
    # explainer.model.eval()
    optimizer = optim.Adam(explainer.elayers.parameters(), lr=args.lr)
    t0 = args.coff_t0
    t1 = args.coff_te
    
    remap = data.remap

    epochs = args.epochs
    nodes = data.nodes
    for epoch in range(epochs):
        tmp = float(t0 * np.power(t1 / t0, epoch / epochs))

        optimizer.zero_grad()
        loss = 0.0
        np.random.shuffle(nodes)
        for nodeid in nodes: # nodeid is not remapped
            if remap:
                nodeid = remap[nodeid]
            
            if args.gnn_task=='node': 
                feature = data.sub_features[nodeid]
                adj = data.sub_adjs[nodeid]
                emb = data.sub_embeds[nodeid]
                label = data.sub_label_tensors[nodeid]
                support = data.sub_support_tensors[nodeid]
                
                with torch.no_grad():
                    output = explainer.model((feature, support), training=False)
                    pred_label = output.argmax(dim=1)

                pred = explainer((feature, adj, 0, emb, tmp), training=True)
                cur_loss = explainer.loss(pred, pred_label, label, 0)
            elif args.gnn_task=='graph':
                feature = torch.tensor(data.feas[nodeid], device=args.device)
                adj = torch.tensor(data.adjs[nodeid], device=args.device)
                emb = data.embs[nodeid]
                label = torch.tensor(data.labels[nodeid])
                pred = explainer((feature, emb, adj, tmp, label.to(args.device)), training=True)
                cur_loss = explainer.loss(pred, label)
            loss += cur_loss
        loss.backward()
        optimizer.step()

        rocauc = test_explainer(explainer ,data)
        print(f'epoch {epoch}, rocauc {rocauc}, loss {loss.item()}')
        

    

if __name__=='__main__':
    """"""
    args.device = 'cuda'
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.task = 'pos'           # 'pos' or 'neg'
    args.gnn_type = 'supervised' # 'supervised' or 'unsupervised'
    args.setting = 1 # default to 1, syn4+neg일때는 3

    ### graph classification ###
    args.dataset = 'BA-2motif'
    args.bn = False     # model
    args.concat = False   # model
    args.setting = 1
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type
    
    ### node classification ###
    # args.dataset = 'syn4'
    # args.bn = True     # model
    # args.concat = True   # model
    # args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type



    ### pos syn3 ###
    # args.epochs = 30
    # args.lr = 0.01
    # args.coff_t0=1.0
    # args.coff_te=1.0
    # args.coff_size = 1
    # args.coff_ent = 0.001
    # args.shift = 0 # sigmoid
    # args.budget = -1

    ### pos syn4 ###
    # args.epochs = 16
    # args.lr = 0.01
    # args.coff_t0=1.0
    # args.coff_te=5.0
    # args.coff_size = 1
    # args.coff_ent = 1
    # args.shift = 0 # sigmoid
    # args.budget = -1

    ### pos ba2motif


    if args.task=='pos':
        args.epochs = 30
        args.lr = 0.01
        args.coff_t0=1.0
        args.coff_te=5.0
        args.coff_size = 1
        args.coff_ent = 1
        args.shift = 0 # sigmoid
        args.budget = -1

    elif args.task=='neg':
        args.lr = 0.001
        args.coff_t0 = 1.0
        args.coff_te = 1.0
        args.coff_size = -10 # neg
        args.coff_ent = 0.01
        args.shift = 3 # sigmoid



    if args.dataset[:3] == 'syn':
        args.gnn_task = 'node'
    else:
        args.gnn_task = 'graph'

    data = SyntheticDataset(args=args)
    data.cuda()
    
    if args.gnn_task =='node':
        from models import GCN_Node as GCN
        model = GCN(args=args, input_dim=data.features.shape[1], output_dim=data.y_train.shape[1]).to(args.device)
        hiddens = [int(s) for s in args.hiddens.split('-')]
        if args.concat: # 이거 explainer 내부로 옮길까
            inputdim = sum(hiddens) * 3 # 양쪽node(edge) + node itself -> 노드3개
        else:
            inputdim = hiddens[-1] * 3
        explainer = PGExplainer_Node(model=model, elayer_inputdim=inputdim, args=args).to(args.device)
        explainer.model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))

        embeds = model.embedding((data.features_tensor, data.support_tensor)).cpu().detach().numpy()
        data.prepare_inductive(embeds=embeds)
        data.cuda() # subgraphs

    elif args.gnn_task =='graph':
        from models import GCN_Graph as GCN
        model = GCN(args=args, input_dim=data.feas.shape[-1], output_dim=data.labels.shape[1]).to(args.device)
        explainer = PGExplainer_Graph(model=model,num_nodes=data.adjs.shape[1], args=args).to(args.device)
        explainer.model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))
        with torch.no_grad():
            data.embs = explainer.model.getNodeEmb((torch.tensor(data.feas,dtype=torch.float32),\
                                torch.tensor(data.adjs,dtype=torch.float32)), training=False)
    
    train_start_time = time.time()
    train_explainer(args, explainer, data)
    train_end_time = time.time()

    infer_start_time = time.time()
    rocauc = explainer.get_explanation_scores(data)
    infer_end_time = time.time()
    print(rocauc)

    train_time = train_end_time - train_start_time
    infer_time = infer_end_time - infer_start_time
    print(f'Training time: {train_time:.2f} seconds')
    print(f'Inference time: {infer_time:.2f} seconds')
