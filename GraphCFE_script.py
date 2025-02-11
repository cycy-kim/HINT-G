import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np

from models import GraphCFE
from dataset import SyntheticDataset
from utils import args

def train_explainer(args, model, explainer, data:SyntheticDataset):
    """
    Train GraphCFE with pretrained GCN model
    """

    epochs = args.epochs
    device = args.device
    save_model = args.save_model
    # y_cf = 1 - np.array(data.labels_all), 실제 출력 아니고 그냥 label 쓰는듯, y_cf_all
    # optimizer = optim.Adam(explainer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(explainer.parameters(), lr=args.lr)
    pred_model = model # 여기에 pretrained model

    allnodes = data.nodes

    if args.gnn_task =='node':
        remap = data.remap
        for epoch in range(epochs + 1):
            explainer.train()

            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0.0, 0.0, 0.0, 0.0, 0.0
            for nodeid in allnodes:
                remapped_nodeid = remap[nodeid]
                
                features = data.sub_features[remapped_nodeid].unsqueeze(0)
                support = data.sub_support_tensors[remapped_nodeid].unsqueeze(0)
                labels = torch.argmax(data.sub_label_tensors[remapped_nodeid].cpu().detach(), axis=1).to(dtype=torch.long, device=device).unsqueeze(0)

                features, support, labels = data.pad_graph((features, support, labels), args.max_num_nodes)
                u = torch.zeros((1, 1), dtype=torch.float32, device=device) # 여기서도 disabled
                y_cf = (1 - labels).unsqueeze(-1) # 아니 근데 이거 ㄹㅇlabel쓰는거 맞나
                
                optimizer.zero_grad()
                model_return = explainer(features, u, support.to_dense(), y_cf)

                # z_cf
                z_mu_cf, z_logvar_cf = explainer.get_represent(model_return['features_reconst'], u, model_return['adj_reconst'], y_cf)

                # compute loss
                loss_params = {'explainer': explainer, 'pred_model': pred_model, 'adj_input': support, 'features_input': features, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
                loss_params.update(model_return)

                loss_results = explainer.compute_loss(loss_params)
                loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'], loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results['loss_kl_cf']
                loss += loss_batch
                loss_kl += loss_kl_batch
                loss_sim += loss_sim_batch
                loss_cfe += loss_cfe_batch
                loss_kl_cf += loss_kl_batch_cf

            # backward propagation
            # loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf

            # cfe로만 학습을 해도 rocauc가 올라가는거같음
            alpha = 5

            if epoch < 450:
                ((loss_sim + loss_kl + 0 * loss_cfe)).backward()
            else:
                ((loss_sim + loss_kl + alpha * loss_cfe)).backward()
            optimizer.step()

            if epoch % 10 == 0:
                rocauc = explainer.get_explanation_scores(data)
                print(f'epoch: {epoch}, loss: {loss_cfe}, rocauc: {rocauc}')
            else:
                print(f'epoch: {epoch}, loss: {loss_cfe}')
    elif args.gnn_task =='graph':
        for epoch in range(epochs + 1):
            explainer.train()

            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0.0, 0.0, 0.0, 0.0, 0.0
            batch_num = 0
            # for batch_idx, data in enumerate(train_loader):
            # for nodeid in allnodes:
            
            batch_size = 32  # Define batch size
            all_batches = [allnodes[i:i + batch_size] for i in range(0, len(allnodes), batch_size)]
            for batch in all_batches:
                batch_num += 1

                features = torch.tensor([data.feas[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
                adj = torch.tensor([data.adjs[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
                u = torch.zeros((len(batch), 1), dtype=torch.float32, device=device)
                labels = torch.tensor([np.argmax(data.labels[nodeid]) for nodeid in batch], dtype=torch.long, device=device)

                y_cf = (1 - labels).unsqueeze(1)

                optimizer.zero_grad()

                # forward pass
                model_return = explainer(features, u, adj, y_cf)

                # z_cf
                z_mu_cf, z_logvar_cf = explainer.get_represent(model_return['features_reconst'], u, model_return['adj_reconst'], y_cf)

                # compute loss
                loss_params = {'explainer': explainer, 'pred_model': pred_model, 'adj_input': adj, 'features_input': features, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
                loss_params.update(model_return)

                loss_results = explainer.compute_loss(loss_params)
                loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'], loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results['loss_kl_cf']
                loss += loss_batch
                loss_kl += loss_kl_batch
                loss_sim += loss_sim_batch
                loss_cfe += loss_cfe_batch
                loss_kl_cf += loss_kl_batch_cf

            # backward propagation
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss / batch_num, loss_kl/batch_num, loss_sim/batch_num, loss_cfe/batch_num, loss_kl_cf/batch_num

            alpha = 5

            if epoch < 450:
                ((loss_sim + loss_kl + 0* loss_cfe)/ batch_num).backward()
            else:
                ((loss_sim + loss_kl + alpha * loss_cfe)/ batch_num).backward()
            optimizer.step()

            if epoch % 10 == 0:
                rocauc = explainer.get_explanation_scores(data)
                print(f'epoch: {epoch}, loss: {loss}, rocauc: {rocauc}')
            else:
                """"""
                # print(f'epoch: {epoch}, loss: {loss}')


if __name__=='__main__':
    """"""
    args.device = 'cuda'
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.task = 'pos'           # 'pos' or 'neg'
    args.gnn_type = 'supervised' # 'supervised' or 'unsupervised'
    args.disable_u = True
    args.setting = 1 # default to 1, syn4+neg일때는 3, syn3+neg도 3?
    # ------------- #

    ### graph classification ###

    ## pos params, epoch 800에서 0.949, setting3인데 1로 해야되나
    # args.dataset = 'BA-2motif'
    # args.hiddens = '20-20-20'
    # args.bn = False     # model
    # args.concat=False   # model
    # args.setting = 1
    # args.model_weight_path = args.save_path + args.dataset # +'_'+ args.gnn_type
    # args.explainer_bn = True
    
    # neg params,  setting (0,100)
    # args.dataset = 'BA-2motif'
    # args.hiddens = '20-20-20'
    # args.bn = False     # model
    # args.concat=False   # model
    # args.setting = 1
    # args.model_weight_path = args.save_path + args.dataset # +'_'+ args.gnn_type
    # args.explainer_bn = True


    ### node classification ###
    args.dataset = 'syn3'
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type
    args.explainer_bn = False # must disable batchnorm for syn's inductive setting


    if args.dataset[:3] == 'syn':
        args.gnn_task = 'node'
    else:
        args.gnn_task = 'graph'

    data = SyntheticDataset(args=args)
    data.cuda()

    if args.gnn_task =='node':
        args.x_dim = data.features_tensor[0].shape[-1] # node
        args.dim_h = 16
        args.dim_z = 16

        from models import GCN_Node as GCN
        model = GCN(args=args, input_dim=data.features.shape[1], output_dim=data.y_train.shape[1]).to(args.device)
        model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))
        embeds = model.embedding((data.features_tensor, data.support_tensor)).cpu().detach().numpy()
        data.prepare_inductive(embeds=embeds)
        data.cuda() # subgraphs

        # for node in data.nodes:
        #     remapped= data.remap[node]
        #     adj = data.sub_adjs[remapped].todense()
        #     sub_edge_labels=data.sub_edge_labels[remapped].todense()
        #     num_node = len(data.sub_adjs[remapped].todense())
        #     sub_adj = torch.tensor(adj, dtype=torch.float32, device='cuda')
        #     # small_sub_adj = sub_adj/10
        #     pred_original = model((data.sub_features[remapped], sub_adj))
        #     # pred_small = model((data.sub_features[remapped], small_sub_adj))
        #     for src in range(num_node):
        #         for des in range(num_node):
        #             # print(sub_edge_labels[src][des])
        #             if not (sub_edge_labels[src,des] or sub_edge_labels[des,src]):
        #                 sub_adj[src][des] = False
        #                 sub_adj[des][src] = False
            
        #     pred_perturb = model((data.sub_features[remapped], sub_adj))

        #     print('original' ,pred_original[0])
        #     print('perturb', pred_perturb[0])
        #     print('------')

        args.max_num_nodes = np.max([sub_support_tensor.size(0) for sub_support_tensor in data.sub_support_tensors]) # node
    
        explainer = GraphCFE(args=args, model=model).to(args.device)

    elif args.gnn_task =='graph':
        args.x_dim = data.feas.shape[-1] # graph
        args.max_num_nodes = data.adjs.shape[-1] # graph
        args.dim_h = 16
        args.dim_z = 16

        from models import GCN_Graph as GCN
        model = GCN(args=args, input_dim=data.feas.shape[-1], output_dim=data.labels.shape[1]).to(args.device)
        explainer = GraphCFE(args=args, model=model).to(args.device)
        explainer.model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))


    train_explainer(args, model, explainer, data)
    rocauc = explainer.get_explanation_scores(data)
    print(rocauc)
