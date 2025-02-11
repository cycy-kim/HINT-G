import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np
from tqdm import tqdm

from models import TAGExplainer
from dataset import SyntheticDataset
from utils import args

"""
tage
1. embed_model(원래꺼에서 linear만 떼면 되니까 그대로 쓰면 될듯)
2. TAGExplainer(embed_model, ...
3. explainer.train_explainer_graph(dataloader)
4. (subgraph, subset), masks = explainer(data, ...)
-> masks가 score

get_subgraph는 필요없음 그냥 SyntheticDataset으로 커버가능

"""
# 클래스 안에 있던거
def train_explainer_graph(self, loader, lr=0.001, epochs=10):
    optimizer = optim.Adam(self.explainer.parameters(), lr=lr)
    for epoch in range(epochs):
        tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / epochs))
        self.model.eval()
        self.explainer.train()
        pbar = tqdm(loader)
        optimizer.zero_grad()
        # data = data.to(self.device)
        # embed, node_embed = self.model(data, emb=True)
        embed = self.model(loader.x.to(self.device), loader.edge_index.to(self.device))
        cond = self.__rand_cond__(1)
        pruned_embed, mask, log = self.explain(loader, embed=embed, 
            condition=cond, tmp=tmp, training=True, batch=loader.batch.to(self.device))

        loss = self.__loss__(embed, pruned_embed, cond, mask)
        pbar.set_postfix({'loss': loss.item(), 'log': log})
        loss.backward()
        optimizer.step()

def train_explainer_batched_graph(args, model, explainer, data:SyntheticDataset):
    """
    Train GraphCFE with pretrained GCN model
    """

    epochs = args.epochs
    device = args.device
    optimizer = optim.Adam(explainer.explainer.parameters(), lr=args.lr)
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
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf

            alpha = 5

            if epoch < 450:
                ((loss_sim + loss_kl + 0 * loss_cfe)).backward()
            else:
                ((loss_sim + loss_kl + alpha * loss_cfe)).backward()
            optimizer.step()

            if epoch % 10 == 0:
                rocauc = explainer.get_explanation_score(data)
                print(f'epoch: {epoch}, loss: {loss}, rocauc: {rocauc}')
            else:
                print(f'epoch: {epoch}, loss: {loss}')
    elif args.gnn_task =='graph':
        for epoch in range(epochs + 1):
            explainer.model.eval()
            explainer.explainer.train()

            tmp = float(args.coff_t0 * np.power(args.coff_te / args.coff_t0, epoch / epochs))
            
            epoch_loss = 0.0
            batch_num = 0
            batch_size = 32  # Define batch size
            all_batches = [allnodes[i:i + batch_size] for i in range(0, len(allnodes), batch_size)]
            for batch in all_batches:
                batch_num += 1

                features = torch.tensor([data.feas[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
                adj = torch.tensor([data.adjs[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
                labels = torch.tensor([np.argmax(data.labels[nodeid]) for nodeid in batch], dtype=torch.long, device=device)
                graph = features, adj, labels
                optimizer.zero_grad()

                # embed = explainer.model(loader.x.to(self.device), loader.edge_index.to(self.device))
                embed = explainer.model((features, adj)) # 원래 쓰던 모델

                # 이거 batch단위로 안되는거같은데..
                cond = explainer.__rand_cond__(1) # 이거 호출하면 좀
                pruned_embed, mask, log = explainer.explain(graph, embed=embed, 
                    condition=cond, tmp=tmp, training=True, batch=graph)

                cur_loss = explainer.__loss__(embed, pruned_embed, cond, mask)
                loss.backward()
                optimizer.step()

                epoch_loss += cur_loss.item()
                
            epoch_loss = epoch_loss / batch_num

            if epoch % 50 == 0:
                rocauc = explainer.get_explanation_score(data)
                print(f'epoch: {epoch}, loss: {epoch_loss}, rocauc: {rocauc}')
            else:
                print(f'epoch: {epoch}, loss: {epoch_loss}')

def train_explainer(args, model, explainer, data:SyntheticDataset):
    """
    Train GraphCFE with pretrained GCN model
    """

    epochs = args.epochs
    device = args.device
    optimizer = optim.Adam(explainer.explainer.parameters(), lr=args.lr)
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
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf

            alpha = 5

            if epoch < 450:
                ((loss_sim + loss_kl + 0 * loss_cfe)).backward()
            else:
                ((loss_sim + loss_kl + alpha * loss_cfe)).backward()
            optimizer.step()

            if epoch % 10 == 0:
                rocauc = explainer.get_explanation_score(data)
                print(f'epoch: {epoch}, loss: {loss}, rocauc: {rocauc}')
            else:
                print(f'epoch: {epoch}, loss: {loss}')
    elif args.gnn_task =='graph':
        for epoch in range(epochs + 1):
            explainer.model.eval()
            explainer.explainer.train()

            tmp = float(args.coff_t0 * np.power(args.coff_te / args.coff_t0, epoch / epochs))
            
            epoch_loss = 0.0
            batch_num = 0
            batch_size = 32  # Define batch size
            all_batches = [allnodes[i:i + batch_size] for i in range(0, len(allnodes), batch_size)]
            for nodeid in allnodes:
            # for batch in all_batches:
                feature = torch.tensor(data.feas[nodeid], dtype=torch.float32, device=device)
                adj = torch.tensor(data.adjs[nodeid], dtype=torch.float32, device=device)
                label = torch.tensor(np.argmax(data.labels[nodeid]), dtype=torch.long, device=device)
                

                # features = torch.tensor([data.feas[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
                # adj = torch.tensor([data.adjs[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
                # labels = torch.tensor([np.argmax(data.labels[nodeid]) for nodeid in batch], dtype=torch.long, device=device)
                graph = feature, adj, label
                optimizer.zero_grad()

                # embed = explainer.model(loader.x.to(self.device), loader.edge_index.to(self.device))
                embed = explainer.model((feature.unsqueeze(0), adj.unsqueeze(0))) # 원래 쓰던 모델

                # 이거 batch단위로 안되는거같은데..
                cond = explainer.__rand_cond__(1) # 이거 호출하면 좀
                pruned_embed, mask, log = explainer.explain(graph, embed=embed, 
                    condition=cond, tmp=tmp, training=True, batch=graph)

                cur_loss = explainer.__loss__(embed, pruned_embed, cond, mask)
                loss.backward()
                optimizer.step()

                epoch_loss += cur_loss.item()
                
            epoch_loss = epoch_loss / batch_num

            if epoch % 50 == 0:
                rocauc = explainer.get_explanation_score(data)
                print(f'epoch: {epoch}, loss: {epoch_loss}, rocauc: {rocauc}')
            else:
                print(f'epoch: {epoch}, loss: {epoch_loss}')


if __name__=='__main__':
    """"""
    args.device = 'cuda'
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.task = 'pos'           # 'pos' or 'neg'
    args.gnn_type = 'supervised' # 'supervised' or 'unsupervised'
    args.setting = 1 # default to 1, syn4+neg일때는 3

    args.embed_dim = 60 # 이거 20*3인가
    args.grad_scale = 0.25
    args.coff_size = 0.05
    args.coff_ent = 0.05
    args.coff_t0 = 1.0
    args.coff_te = 1.0
    # ------------- #

    ### graph classification ###
    args.dataset = 'BA-2motif'
    args.bn = False     # model
    args.concat=False   # model
    args.setting = 1
    args.model_weight_path = args.save_path + args.dataset # +'_'+ args.gnn_type
    args.explainer_bn = True

    
    ### node classification ###
    # args.dataset = 'syn3'
    # args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type
    # args.explainer_bn = False # must disable batchnorm for syn's inductive setting

    if args.dataset[:3] == 'syn':
        args.gnn_task = 'node'
    else:
        args.gnn_task = 'graph'

    data = SyntheticDataset(args=args)
    data.cuda()

    if args.gnn_task =='node':
        args.x_dim = data.sub_features[0].shape[-1] # node
        args.max_num_nodes = np.max([sub_support_tensor.size(0) for sub_support_tensor in data.sub_support_tensors]) # node

        from models import GCN_Node as GCN
        model = GCN(args=args, input_dim=data.features.shape[1], output_dim=data.y_train.shape[1]).to(args.device)
        explainer = TAGExplainer(args=args, model=model).to(args.device)
        explainer.model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))

    elif args.gnn_task =='graph':
        args.x_dim = data.feas.shape[-1] # graph
        args.max_num_nodes = data.adjs.shape[-1] # graph

        from models import GCN_Graph as GCN
        model = GCN(args=args, input_dim=data.feas.shape[-1], output_dim=data.labels.shape[1]).to(args.device)
        explainer = TAGExplainer(args=args, model=model).to(args.device)
        explainer.model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))

    train_explainer(args, model, explainer, data)
    rocauc = explainer.get_explanation_scores(data)
    print(rocauc)
