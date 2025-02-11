import itertools
import copy
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.functional import cross_entropy
from torch.autograd import grad
from torch_geometric.nn import DenseGCNConv, DenseGraphConv
from torch_geometric.nn import global_max_pool, global_mean_pool

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from scipy.sparse import coo_matrix
import networkx as nx

from .models import MLP, GCN_Node, GCN_Graph
from .layers import GraphConvolution_Node, GraphConvolution_Graph
from dataset import Dataset, SyntheticDataset, MutagDataset

from utils import *


class Explainer(nn.Module):
    def __init__(self):
        """"""
        super(Explainer, self).__init__()

    def _cuda(self):
        """"""
    
    # def _prepare_model(self):
    #     """"""

    @torch.no_grad()
    def _pred_changes(self, model, data, nodeid, edge, original_output, target_nodeid=0):
        """If the predication actually flips when edge(param) is added.

        Args:
            model (_type_): 
            data + nodeid: one subgraph / tree
            edge (_type_): edge to add
            original_output (_type_): 
            target_nodeid (int, optional): _description_. Defaults to 0.

        Returns:
            Boolean tensor: _description_
        """
        model.eval() ###
        try:
            device = self.device
        except Exception as e:
            device = self.args.device
        src, des = edge
        
        remapped_id = data.remap[nodeid]
        sub_feature_tensor = data.sub_features[remapped_id]
        coalesced_tensor = data.sub_support_tensors[remapped_id].coalesce()
        
        indices = coalesced_tensor.indices()
        values = coalesced_tensor.values()
        size = coalesced_tensor.size()
        
        new_edges = torch.tensor([[src, des], [des, src]], device=device)
        new_values = torch.tensor([1.0, 1.0], device=device)
        
        new_indices = torch.cat([indices, new_edges], dim=1)
        new_values = torch.cat([values, new_values])
        
        perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
        perturbed_output = model((sub_feature_tensor, perturbed_support_tensor), training=False)
        
        original_pred = torch.argmax(original_output, dim=1)[target_nodeid]
        perturbed_pred = torch.argmax(perturbed_output, dim=1)[target_nodeid]
        # original_pred = torch.argmax(original_output, dim=1)
        # perturbed_pred = torch.argmax(perturbed_output, dim=1)

        # if torch.any(original_pred == 1):
        #     return False
        
        model.eval() ###
        return not torch.equal(original_pred, perturbed_pred)


    # @torch.no_grad()
    # def _pred_changes(self, model, data, nodeid, edge, original_output, target_nodeid=0):
    #     model.eval()

    #     try:
    #         device = self.device
    #     except Exception as e:
    #         device = self.args.device

    #     src, des = edge
    #     remapped_id = data.remap[nodeid]
    #     sub_feature_tensor = data.sub_features[remapped_id]
    #     coalesced_tensor = data.sub_support_tensors[remapped_id].coalesce()

    #     indices = coalesced_tensor.indices()
    #     values = coalesced_tensor.values()
    #     size = coalesced_tensor.size()

    #     new_edges = torch.tensor([[src, des], [des, src]], device=device)
    #     new_values = torch.tensor([1.0, 1.0], device=device)

    #     new_indices = torch.cat([indices, new_edges], dim=1)
    #     new_values = torch.cat([values, new_values])

    #     perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
    #     perturbed_output = model((sub_feature_tensor, perturbed_support_tensor), training=False)

    #     original_pred = torch.argmax(original_output, dim=1)
    #     perturbed_pred = torch.argmax(perturbed_output, dim=1)

    #     G = nx.Graph()
    #     edge_list = list(zip(indices[0].cpu().numpy(), indices[1].cpu().numpy()))
    #     G.add_edges_from(edge_list)
    #     G.add_edge(src, des)

    #     try:
    #         cycle = nx.find_cycle(G, source=src)
    #         cycle_nodes = set(node for edge in cycle for node in edge)
    #     except nx.NetworkXNoCycle:
    #         return False

    #     original_cycle_preds = original_pred[list(cycle_nodes)]
    #     perturbed_cycle_preds = perturbed_pred[list(cycle_nodes)]

    #     if torch.any(original_pred == 1):
    #         return False

    #     # return torch.any(original_cycle_preds != perturbed_cycle_preds)
    #     return torch.all(perturbed_cycle_preds == 1)

    def _get_pos_reals(self, nodeid, score_edges, data:Dataset=None, return_topk=False):
        """
        이거 좀 효율적으로..
        """
        if data is None:
            data = self.data
        edge_reals = {}
        topk=0
        if self.args.gnn_task == 'node':
            sub_edge_label = data.sub_edge_labels[data.remap[nodeid]].todense()
            for _, edge in score_edges:
                src, des = edge
                if sub_edge_label[src,des] or sub_edge_label[des,src]:
                    edge_reals[(src,des)] = 1
                    topk += 1
                else:
                    edge_reals[(src,des)] = 0
        else:
            if self.args.dataset == 'BA-2motif':
                for _, edge in score_edges:
                    src, des = edge
                    if src>=20 and src<25 and des>=20 and des<25:
                        edge_reals[(src,des)] = 1
                        topk += 1
                    else:
                        edge_reals[(src,des)] = 0

            if self.args.dataset == 'Mutagenicity':
                edge_list = data.edge_lists[nodeid]
                edge_labels = data.edge_label_lists[nodeid]
                edge_labels_dict = {}
                for edge, l in list(zip(edge_list,edge_labels)):
                    edge_labels_dict[edge] = l
                    topk += 1
                for _, edge in score_edges:
                    edge_reals[edge] = edge_labels_dict[edge]
        
        if return_topk:
            return edge_reals, topk
        else:
            return edge_reals

    def _get_sub_label(self, nodeid):
        """
        For syn3 or syn4

        sub_labels는 안정확하고
        sub_edge_labels는 정확해서
        sub_edge_labels로 sub_labels 계산하기

        Args:
            explainer (Explainer): XAIFG model
            nodeid (_type_): _description_
        """
        
        data = self.data
        sub_edge_label = data.sub_edge_labels[data.remap[nodeid]].todense()
        num_node = len(sub_edge_label)
        
        sub_label = torch.zeros(num_node)

        for src in range(num_node):
            for des in range(num_node):
                if sub_edge_label[src,des] or sub_edge_label[des,src]:
                    sub_label[src] = True
                    sub_label[des] = True
        
        
        return sub_label

    def _get_neg_reals(self, nodeid, score_edges, data:Dataset=None, return_topk=False):
        """"""
        if data is None:
            data = self.data
        model = self.model
        edge_reals = {}
        topk=0
        gnn_type = self.args.gnn_type
        
        if gnn_type == 'unsupervised':
            # unsup node neg만 homophily
            if self.args.gnn_task == 'node' and self.args.task == 'neg':
                """"""
                # sub_edge_label = data.sub_edge_labels[data.remap[nodeid]].todense()
                # sub_node_label = np.argmax(data.sub_labels[data.remap[nodeid]], axis=-1)
                sub_node_label = self._get_sub_label(nodeid)
                for _, edge in score_edges:
                    src, des = edge
                    if sub_node_label[src] != sub_node_label[des]: # homophily
                        edge_reals[(src,des)] = 1
                        topk+=1
                    else:
                        edge_reals[(src,des)] = 0

        else:
            if self.args.dataset == 'syn3':
                """
                pred change
                """
                with torch.no_grad():
                    original_output = model((data.sub_features[data.remap[nodeid]], data.sub_support_tensors[data.remap[nodeid]]))
                for _, edge in score_edges:
                    src, des = edge
                    # 이거 좀더 효율적으로
                    is_pred_changed = self._pred_changes(model, data, nodeid, edge, original_output, target_nodeid=0)
                    if is_pred_changed:
                        edge_reals[(src,des)] = 1
                        topk+=1
                    else:
                        edge_reals[(src,des)] = 0
                # print(edge_reals)
            elif self.args.dataset == 'syn4':
                sub_edge_label = data.sub_edge_labels[data.remap[nodeid]].todense()
                for _, edge in score_edges:
                    src, des = edge
                    if sub_edge_label[src,des] or sub_edge_label[des,src]:
                        edge_reals[(src,des)] = 1
                        topk+=1
                    else:
                        edge_reals[(src,des)] = 0
            elif self.args.dataset == 'BA-2motif':
                for _, edge in score_edges:
                    src, des = edge
                    if src>=20 and src<25 and des>=20 and des<25:
                        edge_reals[(src,des)] = 1
                        topk+=1
                    else:
                        edge_reals[(src,des)] = 0


        if return_topk:
            return edge_reals, topk
        else:
            return edge_reals

    def _pos_singlegraph(self, nodeid):
        """_summary_

        Args:
            nodeid (_type_): not remapped, original nodeid

        Raises:
            NotImplementedError
        
        Returns:
            score_edges(List): (score, edge)
            score(float): 
            edge(tuple): (src, des)
        """
        raise NotImplementedError
    
    def _neg_singlegraph(self, nodeid):
        """_summary_

        Args:
            nodeid (_type_): not remapped, original nodeid

        Raises:
            NotImplementedError
        
        Returns:
            score_edges(List): (score, edge)
                score(float): 
                edge(tuple): (src, des)
        """
        raise NotImplementedError

    def explain(self):
        """"""

    def explain_score(self):
        """"""



class XAIFG(Explainer):
    def __init__(self, args):
        super(XAIFG, self).__init__()

        self.args = args
        if args.dataset == 'Mutagenicity':
            self.data = MutagDataset(args=args)
        else:
            self.data = SyntheticDataset(args=args)
        self.model = self._prepare_model()
        if self.args.gnn_task == 'node':
            if self.args.gnn_type =='supervised':
                embeds = self.model.embedding((self.data.features_tensor, self.data.support_tensor)).cpu().detach().numpy()
            else:
                dummy_embeds = np.zeros(self.data.features.shape[0])
                embeds = dummy_embeds
            self.data.prepare_inductive(embeds=embeds)
            
        if args.gnn_type=='supervised':
            self.criterion = nn.CrossEntropyLoss()
        elif args.gnn_type=='unsupervised':
            self.criterion = nn.BCEWithLogitsLoss()
        self._cuda()

    def _cuda(self):
        """
        model
        dataset
        """
        device = self.args.device
        self.model = self.model.to(device)
        self.data.cuda()

    @torch.no_grad()
    def _test_model(self):
        """Check if the model works properly in the inductive setting."""
        model = self.model
        data = self.data
        model.eval()
        preds = []
        reals = []
        pred_scores = []
        target_nodeid = 0
        for node in data.nodes:
            output = model((data.sub_features[data.remap[node]], data.sub_support_tensors[data.remap[node]]))
            pred = torch.argmax(output, dim=1)
            real = np.argmax(data.sub_labels[data.remap[node]])

            pred_scores.append(output[:,1][target_nodeid].cpu())
            preds.append(pred[target_nodeid].cpu())
            reals.append(real)
        
        acc = accuracy_score(reals, preds)
        try:
            roc_auc = roc_auc_score(reals, pred_scores)
        except Exception as e:
            roc_auc = 0 
        print(acc, roc_auc)
        return acc, roc_auc
    
    def _prepare_model(self):
        data = self.data
        model_weight_path = self.args.model_weight_path
        
        # 개구린데..
        if self.args.gnn_task == 'node':
            if self.args.gnn_type == 'supervised':
                from models import GCN_Node as GCN
                model = GCN(args=self.args, input_dim=data.features.shape[1], output_dim=data.y_train.shape[1])
                model.load_state_dict(torch.load(model_weight_path, weights_only=True))
            if self.args.gnn_type == 'unsupervised':
                from torch_geometric.nn import GCN
                model = GCN(data.features.shape[1], hidden_channels=self.args.hidden_dim, num_layers=3)
                model.load_state_dict(torch.load(model_weight_path, weights_only=True))

        elif self.args.gnn_task == 'graph':
            from models import GCN_Graph as GCN
            model = GCN(args=self.args, input_dim=data.feas.shape[-1], output_dim=data.labels.shape[1])
            model.load_state_dict(torch.load(model_weight_path, weights_only=True))

        model.eval()
        return model

    def _find_k_hops(self, adj, nodes, hops):
        indices = adj.coalesce().indices()
        current_level = set(nodes)
        visited = set(nodes)
        next_level = set()

        for _ in range(hops):
            for n in current_level:
                neighbors = indices[1][indices[0] == n].tolist()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
            
            current_level = next_level
            next_level = set()

            if not current_level:
                break
        
        return list(visited)

    def _hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem) # 이게 원래꺼
            # element_product += torch.mean(grad_elem * v_elem)
        
        return_grads = grad(element_product,model_params,create_graph=True)
        return return_grads
           
    def _gif_approxi(self, grads, nodeid, losses):
        # 이거 args에서?
        iteration = self.args.iter
        scale = self.args.scale
        damp = 0
        grad_all, grad1, grad2 = grads
        loss, loss1, loss2 = losses
        original_params = [p.data for p in self.model.parameters() if p.requires_grad]
        v = tuple(grad1_t - grad2_t for grad1_t, grad2_t in zip(grad1,grad2))
        h_estimate = tuple(grad1_t - grad2_t for grad1_t, grad2_t in zip(grad1,grad2))
        for _ in range(iteration):
            model_params  = [p for p in self.model.parameters() if p.requires_grad]
            hv            = self._hvps(grad_all, model_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]
        
        params_change = [h_est * v1 for v1, h_est in zip(v, h_estimate)]

        # score = loss2
        # print(len(original_params))
        # for o_param in original_params:
        #     print(o_param.shape)
        # exit()
        # score = torch.mean(torch.stack([
        #     torch.mean(torch.abs(t)/(torch.abs(w)))
        #     for w, t in zip(original_params, params_change) if w.dim() == 1
        #     ]))
        score = torch.mean(torch.stack([torch.mean(torch.abs(t)/(torch.abs(w)+0.001)) for w, t in zip(original_params, params_change)]))
        return score

    def _pos_singlegraph(self, nodeid):
        data = self.data
        model = self.model
        dataset_name = self.args.dataset
        score_edges = []
        
        if self.args.gnn_task=='node' and self.args.gnn_type=='supervised':
            original_feature_tensor = data.sub_features[data.remap[nodeid]]
            original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
            label = data.sub_label_tensors[data.remap[nodeid]]
            indices = original_support_tensor.coalesce().indices()
            values = original_support_tensor.coalesce().values()
            size = original_support_tensor.coalesce().size()
        
            influenced_hops = len([int(s) for s in self.args.hiddens.split('-')])
            
            for edge_index in range(len(values)):
                src, des = indices[0][edge_index].item(), indices[1][edge_index].item()
                if src >= des:
                    continue
                
                mask = ~(((indices[0] == src) & (indices[1] == des)) | ((indices[0] == des) & (indices[1] == src)))
                # print(mask)
                new_indices = indices[:, mask]
                new_values = values[mask]
                perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
                original_output = model((original_feature_tensor, original_support_tensor), training=False)
                perturbed_output = model((original_feature_tensor, perturbed_support_tensor), training=False)


                influenced_nodes = self._find_k_hops(perturbed_support_tensor, [src, des], influenced_hops)
                # influenced_nodes = self._find_k_hops(perturbed_support_tensor, [0], influenced_hops)
                # influenced_nodes = [0]

                mask1 = np.array([False] * perturbed_output.shape[0])
                mask1[influenced_nodes] = True
                
                mask2 = mask1

                original_whole_output = model((data.features_tensor, data.support_tensor))
                all_label_tensor = torch.tensor(data.all_label, dtype=torch.float32, device=self.args.device)

                loss = self.criterion(original_whole_output[data.train_mask_tensor], all_label_tensor[data.train_mask_tensor])
                loss1 = self.criterion(original_output[mask1], label[mask1])
                loss2 = self.criterion(perturbed_output[mask2], label[mask2])
                model_params = [p for p in model.parameters() if p.requires_grad]
                
                grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                score_edges.append((score.item(), (src, des)))
            score_edges.sort(reverse=True, key=lambda x: x[0])
        
        elif self.args.gnn_task=='node' and self.args.gnn_type=='unsupervised':
            original_sub_feature_tensor = data.sub_features[data.remap[nodeid]]
            original_sub_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
            indices = original_sub_support_tensor.coalesce().indices()
            values = original_sub_support_tensor.coalesce().values()
            size = original_sub_support_tensor.coalesce().size()
            num_node = size[0]

            original_sub_edge_index = original_sub_support_tensor.to_dense().nonzero(as_tuple=False).t()

            ## label
            row = torch.arange(num_node).repeat(num_node)
            col = torch.arange(num_node).unsqueeze(1).repeat(1, num_node).view(-1)
            sub_edge_index_all = torch.stack([row, col], dim=0) 
            
            whole_num_node = data.support_tensor.coalesce().size()[0]
            whole_row = torch.arange(whole_num_node).repeat(whole_num_node)
            whole_col = torch.arange(whole_num_node).unsqueeze(1).repeat(1, whole_num_node).view(-1)
            edge_index_whole = torch.stack([whole_row, whole_col], dim=0) 
            
            sub_edge_label_tensor = torch.zeros(sub_edge_index_all.size(1), dtype=torch.float32, device=self.args.device)
            src = original_sub_edge_index[0]
            dst = original_sub_edge_index[1]
            all_indices = src * num_node + dst
            sub_edge_label_tensor[all_indices] = 1

            whole_edge_index = data.support_tensor.to_dense().nonzero(as_tuple=False).t()
            full_edge_label_tensor = torch.zeros(edge_index_whole.size(1), dtype=torch.float32, device=self.args.device)
            src = whole_edge_index[0]
            dst = whole_edge_index[1]
            all_indices = src * whole_num_node + dst
            full_edge_label_tensor[all_indices] = 1

            for edge_index in range(len(values)):
                src, des = indices[0][edge_index].item(), indices[1][edge_index].item()
                if src >= des:
                    continue
                # sub_num_nodes = original_sub_feature_tensor.size(0)
                mask = ~(((indices[0] == src) & (indices[1] == des)) | ((indices[0] == des) & (indices[1] == src)))
                new_indices = indices[:, mask]
                new_values = values[mask]
                perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
                
                perturbed_edge_index = perturbed_support_tensor.to_dense().nonzero(as_tuple=False).t()

                h_ori = model(original_sub_feature_tensor, original_sub_edge_index)
                h_ori_src = h_ori[sub_edge_index_all[0]]
                h_ori_dst = h_ori[sub_edge_index_all[1]]
                pred_ori = (h_ori_src * h_ori_dst).sum(dim=-1)

                h_per = model(original_sub_feature_tensor, perturbed_edge_index)
                h_per_src = h_per[sub_edge_index_all[0]]
                h_per_dst = h_per[sub_edge_index_all[1]]
                pred_per = (h_per_src * h_per_dst).sum(dim=-1)
                
                loss = self.criterion(pred_ori, sub_edge_label_tensor)
                loss1 = self.criterion(pred_ori, sub_edge_label_tensor)
                loss2 = self.criterion(pred_per, sub_edge_label_tensor)
                
                model_params = [p for p in model.parameters() if p.requires_grad]
                
                grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                
                # adj = perturbed_support_tensor.to_dense()
                # # degree_sum = max(torch.sum(adj[src]) , torch.sum(adj[des]))
                # degree_sum = abs(torch.sum(adj[src]) - torch.sum(adj[des]))
                # # exit() 
                # # print(score)
                # score *= degree_sum # 양쪽 노드 degree
                
                score_edges.append((score.item(), (src, des)))

            score_edges.sort(reverse=True, key=lambda x: x[0])
            # print(score_edges)
        
        elif self.args.gnn_task=='graph' and self.args.gnn_type=='supervised':
            if dataset_name == 'BA-2motif' or dataset_name == 'Mutagenicity':
                adj, fea, label = data.adjs[nodeid], data.feas[nodeid], data.labels[nodeid]
                graph = adj, fea, label
                label = torch.tensor(label)
                fea_tensor = torch.tensor(fea, dtype=torch.float32).unsqueeze(0)
                label_tensor = torch.argmax(label).unsqueeze(0).to(args.device)
                num_nodes = adj.shape[-1]
                for src in range(num_nodes):
                    for des in range(num_nodes):
                        if src >= des or not adj[src][des]:
                            continue
                        
                        new_adj = copy.deepcopy(adj)
                        new_adj[src][des] = False
                        new_adj[des][src] = False
                        # new_graph = (new_adj, fea, label)
                        
                        original_output = model((fea_tensor, torch.tensor(adj, dtype=torch.float32).unsqueeze(0)))
                        perturbed_output = model((fea_tensor, torch.tensor(new_adj, dtype=torch.float32).unsqueeze(0)))
                        
                        # print(original_output)
                        # print(perturbed_output)
                        # print(label_tensor)
                        # if torch.argmax(original_output, dim=-1).item() != torch.argmax(perturbed_output, dim=-1).item():
                        #     print('pred changed')
                        # print('-----')
                        
                        loss = cross_entropy(original_output, label_tensor)
                        loss1 = cross_entropy(original_output, label_tensor)    
                        loss2 = cross_entropy(perturbed_output, label_tensor)

                        model_params = [p for p in model.parameters() if p.requires_grad]
                        
                        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                        # score = gif_approxi((grad_all, grad1, grad2), graph)
                        score = self._gif_approxi((grad_all, grad1, grad2), graph, (loss, loss1, loss2))
                        score_edges.append((score.item(), (src, des)))
                score_edges.sort(reverse=True, key=lambda x: x[0])
                
        elif self.args.gnn_task=='graph' and self.args.gnn_type=='unsupervised':
            """
            Not implemented yet (no appropriate unsupervised setting for graph classification).
            """

        return score_edges

    def _neg_singlegraph(self, nodeid):
        data = self.data
        model = self.model
        dataset_name = self.args.dataset
        score_edges= []


        if self.args.gnn_task=='node' and self.args.gnn_type=='supervised':        
            if dataset_name == 'syn3':
                original_feature_tensor = data.sub_features[data.remap[nodeid]]
                original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
                label = data.sub_label_tensors[data.remap[nodeid]]
                
                indices = original_support_tensor.coalesce().indices()
                values = original_support_tensor.coalesce().values()
                size = original_support_tensor.coalesce().size()
                num_node = size[0]
                
                influenced_hops = len([int(s) for s in self.args.hiddens.split('-')]) + 1
                # influenced_hops = 1

                for src in range(num_node):
                    for des in range(num_node):
                        if ((indices[0] == src) & (indices[1] == des)).any().item() or (src>=des):
                            continue

                        new_edge1 = torch.tensor([[src], [des]], device=indices.device)
                        new_edge2 = torch.tensor([[des], [src]], device=indices.device)
                        new_value = torch.tensor([1.0], device=values.device)
                        
                        new_indices = torch.cat([indices, new_edge1, new_edge2], dim=1)
                        new_values = torch.cat([values, new_value, new_value])
                        
                        perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
                        original_output = model((original_feature_tensor, original_support_tensor), training=False) # 이거 grad 계산해서 왠지 매번 해야될거같은 느낌
                        perturbed_output = model((original_feature_tensor, perturbed_support_tensor), training=False)

                        influenced_nodes = self._find_k_hops(perturbed_support_tensor, [src, des], influenced_hops)
                        
                        mask1 = np.array([False] * perturbed_output.shape[0])
                        mask1[influenced_nodes] = True
                        # mask1[0] = True # target node=0

                        mask2 = mask1

                        original_whole_output = model((data.features_tensor, data.support_tensor))
                        all_label_tensor = torch.tensor(data.all_label, dtype=torch.float32, device=self.args.device)

                        loss = self.criterion(original_whole_output[data.train_mask_tensor], all_label_tensor[data.train_mask_tensor])
                        loss1 = self.criterion(original_output[mask1], label[mask1])
                        loss2 = self.criterion(perturbed_output[mask2], label[mask2])

                        model_params = [p for p in model.parameters() if p.requires_grad]
                        
                        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                        score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                        score_edges.append((score.item(), (src, des)))
                score_edges.sort(reverse=True, key=lambda x: x[0])
                return score_edges
                
            elif dataset_name == 'syn4':
                """"""
                original_feature_tensor = data.sub_features[data.remap[nodeid]]
                original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
                label = data.sub_label_tensors[data.remap[nodeid]]
                # label = torch.tensor(data.sub_labels[data.remap[nodeid]])
                indices = original_support_tensor.coalesce().indices()
                values = original_support_tensor.coalesce().values()
                size = original_support_tensor.coalesce().size()
                num_node = size[0]
                
                influenced_hops = len([int(s) for s in self.args.hiddens.split('-')])
                
                deletion_candidates = []
                for edge_index in range(len(values)):
                    src, des = indices[0][edge_index].item(), indices[1][edge_index].item()
                    if src >= des:
                        continue
                    if data.sub_edge_labels[data.remap[nodeid]].todense()[src,des] or data.sub_edge_labels[data.remap[nodeid]].todense()[des,src]:
                        deletion_candidates.append((src, des))
                    
                # edges_to_delete = random.sample(deletion_candidates, k)
                edges_to_delete = random.sample(deletion_candidates, 1) # 일단 k=1 고정

                for src_del, des_del in edges_to_delete:
                    mask = ~(((indices[0] == src_del) & (indices[1] == des_del)) | ((indices[0] == des_del) & (indices[1] == src_del)))
                    new_indices = indices[:, mask]
                    new_values = values[mask]
                    del_original_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)

                    num_layers = len([int(s) for s in self.args.hiddens.split('-')])
                    k_hop_nodes_from_target = self._find_k_hops(del_original_support_tensor, [0], num_layers+1)
                    edges_to_add = list(itertools.combinations(k_hop_nodes_from_target, 2))
                    
                    for edge_to_add in edges_to_add:
                        # addition
                        src, des = edge_to_add
                        if ((new_indices[0] == src) & (new_indices[1] == des)).any().item() or (src>=des):
                            continue
                        new_edge1 = torch.tensor([[src], [des]], device=self.args.device)
                        new_edge2 = torch.tensor([[des], [src]], device=self.args.device)
                        new_value = torch.tensor([1.0], device=self.args.device)
                        
                        added_indices = torch.cat([new_indices, new_edge1, new_edge2], dim=1)
                        added_values = torch.cat([new_values, new_value, new_value])
                        
                        perturbed_support_tensor = torch.sparse_coo_tensor(added_indices, added_values, size)
                        original_output = model((original_feature_tensor, del_original_support_tensor), training=False)
                        perturbed_output = model((original_feature_tensor, perturbed_support_tensor), training=False)

                        influenced_nodes = self._find_k_hops(perturbed_support_tensor, [src, des], influenced_hops)
                        mask1 = np.array([False] * perturbed_output.shape[0])
                        mask1[influenced_nodes] = True
                        
                        mask2 = mask1

                        original_whole_output = model((data.features_tensor, data.support_tensor))
                        all_label_tensor = torch.tensor(data.all_label, dtype=torch.float32, device=self.args.device)
                        
                        loss = self.criterion(original_whole_output[data.train_mask_tensor], all_label_tensor[data.train_mask_tensor])
                        loss1 = self.criterion(original_output[mask1], torch.zeros(original_output.size(0), dtype=torch.long, device=self.args.device)[mask1])
                        loss2 = self.criterion(perturbed_output[mask2], torch.zeros(perturbed_output.size(0), dtype=torch.long, device=self.args.device)[mask2])

                        model_params = [p for p in model.parameters() if p.requires_grad]

                        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                        score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                        score_edges.append((score.item(), (src, des)))
        
        elif self.args.gnn_task=='node' and self.args.gnn_type=='unsupervised':
            """"""
            original_sub_feature_tensor = data.sub_features[data.remap[nodeid]]
            original_sub_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
            indices = original_sub_support_tensor.coalesce().indices()
            values = original_sub_support_tensor.coalesce().values()
            size = original_sub_support_tensor.coalesce().size()
            num_node = size[0]

            original_sub_edge_index = original_sub_support_tensor.to_dense().nonzero(as_tuple=False).t()

            ## label
            row = torch.arange(num_node).repeat(num_node)
            col = torch.arange(num_node).unsqueeze(1).repeat(1, num_node).view(-1)
            sub_edge_index_all = torch.stack([row, col], dim=0) 
            
            # whole_num_node = data.support_tensor.coalesce().size()[0]
            # whole_row = torch.arange(whole_num_node).repeat(whole_num_node)
            # whole_col = torch.arange(whole_num_node).unsqueeze(1).repeat(1, whole_num_node).view(-1)
            # whole_edge_index_all = torch.stack([whole_row, whole_col], dim=0).to(self.args.device)
            # edge_index_whole = data.support_tensor.coalesce().indices()
            
            sub_edge_label_tensor = torch.zeros(sub_edge_index_all.size(1), dtype=torch.float32, device=self.args.device)
            src = original_sub_edge_index[0]
            dst = original_sub_edge_index[1]
            all_indices = src * num_node + dst
            sub_edge_label_tensor[all_indices] = 1

            # whole_edge_index = data.support_tensor.to_dense().nonzero(as_tuple=False).t()
            # full_edge_label_tensor = torch.zeros(whole_num_node*whole_num_node, dtype=torch.float32, device=self.args.device)
            # src = whole_edge_index[0]
            # dst = whole_edge_index[1]
            # all_indices = src * whole_num_node + dst
            # full_edge_label_tensor[all_indices] = 1
            for src in range(num_node):
                for des in range(num_node):
                    if ((indices[0] == src) & (indices[1] == des)).any().item() or (src>=des):
                        continue
                    
                    new_edge1 = torch.tensor([[src], [des]], device=indices.device)
                    new_edge2 = torch.tensor([[des], [src]], device=indices.device)
                    new_value = torch.tensor([1.0], device=values.device)
                    
                    new_indices = torch.cat([indices, new_edge1, new_edge2], dim=1)
                    new_values = torch.cat([values, new_value, new_value])
                    
                    perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
                    perturbed_edge_index = perturbed_support_tensor.to_dense().nonzero(as_tuple=False).t()

                    # whole
                    # h_whole = model(data.features_tensor, edge_index_whole)
                    # h_whole_src = h_whole[whole_edge_index_all[0]]
                    # h_whole_dst = h_whole[whole_edge_index_all[1]]
                    # pred_whole = (h_whole_src * h_whole_dst).sum(dim=-1)
                    
                    # original
                    h_ori = model(original_sub_feature_tensor, original_sub_edge_index)
                    h_ori_src = h_ori[sub_edge_index_all[0]]
                    h_ori_dst = h_ori[sub_edge_index_all[1]]
                    pred_ori = (h_ori_src * h_ori_dst).sum(dim=-1)
                    
                    # perturb
                    h_per = model(original_sub_feature_tensor, perturbed_edge_index)
                    h_per_src = h_per[sub_edge_index_all[0]]
                    h_per_dst = h_per[sub_edge_index_all[1]]
                    pred_per = (h_per_src * h_per_dst).sum(dim=-1)
                    
                    # loss = self.criterion(pred_whole, full_edge_label_tensor)
                    loss = self.criterion(pred_ori, sub_edge_label_tensor)
                    loss1 = self.criterion(pred_ori, sub_edge_label_tensor)
                    loss2 = self.criterion(pred_per, sub_edge_label_tensor)
                    
                    model_params = [p for p in model.parameters() if p.requires_grad]
                    
                    grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                    grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                    grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                    score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                    
                    # adj = perturbed_support_tensor.to_dense()
                    # d_src = torch.sum(adj[src])
                    # d_des = torch.sum(adj[des])
                    # degree_sum = min(d_src, d_des)
                    # # degree_sum = d_src+ d_des
                    # score /= degree_sum # 양쪽 노드 degree

                    score_edges.append((score.item(), (src, des)))
                    
        elif dataset_name == 'BA-2motif':
            adj, fea, label = data.adjs[nodeid], data.feas[nodeid], data.labels[nodeid]
            graph = adj, fea, label
            label = torch.tensor(label, device=args.device)

            for src in range(0,25):
                for des in range(0,25):
                    if src >= des or adj[src][des]:
                        continue

                    new_adj = copy.deepcopy(adj)
                    new_adj[src][des] = True
                    new_adj[des][src] = True
                    new_graph = (new_adj, fea, label)
                    
                    original_output = model((torch.tensor(fea).unsqueeze(0), torch.tensor(adj).float().unsqueeze(0)))
                    perturbed_output = model((torch.tensor(fea).unsqueeze(0), torch.tensor(new_adj).float().unsqueeze(0)))
                        
                    label_tensor = torch.argmax(label.clone().detach()).unsqueeze(0)
                    loss = cross_entropy(original_output, label_tensor)
                    loss1 = cross_entropy(original_output, label_tensor)    
                    loss2 = cross_entropy(perturbed_output, label_tensor)

                    model_params = [p for p in model.parameters() if p.requires_grad]
                    # create graph 있어야되나 -> hvp구할때 필요??
                    grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                    grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                    grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                    score = self._gif_approxi((grad_all, grad1, grad2), graph, (loss, loss1, loss2))
                    score_edges.append((score.item(), (src, des)))

        score_edges.sort(reverse=True, key=lambda x: x[0])
        return score_edges
    
    def edge_influences(self):
        auc_rocs = []
        acc_list = []
        precision_list = []
        recall_list = []

        allnodes = self.data.nodes
        for node in tqdm(allnodes):
        # for node in allnodes:
            if self.args.task == 'neg':
                score_edges = self._neg_singlegraph(node)
                edge_reals, topk = self._get_neg_reals(node, score_edges, return_topk=True)
            elif self.args.task == 'pos':
                score_edges = self._pos_singlegraph(node)
                edge_reals, topk = self._get_pos_reals(node, score_edges, return_topk=True)
            
            # print(score_edges)
            all_reals = [1 if edge_reals[edge] else 0 for _, edge in score_edges]
            topk_scores = score_edges[:topk]
            topk_edges = {edge for _, edge in topk_scores}
            preds = [1 if edge in topk_edges else 0 for _, edge in score_edges]

            acc = accuracy_score(all_reals, preds)
            precision = precision_score(all_reals, preds, zero_division=0)
            recall = recall_score(all_reals, preds, zero_division=0)

            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            try:
                scores = [score for score, _ in score_edges]
                rocauc = roc_auc_score(all_reals, scores)
                auc_rocs.append(rocauc)
                print(f'rocauc: {rocauc}')
                # print(f'acc: {acc}')
                # print(f'precision: {precision}')
                # print(f'recall: {recall}')

                # remapped_node = self.data.remap[node]
                # adj = self.data.sub_support_tensors[remapped_node].cpu().to_dense()
                # sub_node_label = self._get_sub_label(node)
                # print(score_edges)
                # print(len(edge_reals))
                # print(sub_node_label)
                # print(topk)
                # plot_adj(adj, show=True, save=False, graph_index=True)
            
            except Exception as e:
                print(e)
                continue

        mean_acc = np.mean(acc_list)
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_auc_roc = np.mean(auc_rocs)

        print(f"Mean Accuracy: {mean_acc:.2f}")
        print(f"Mean Precision: {mean_precision:.2f}")
        print(f"Mean Recall: {mean_recall:.2f}")
        print(f"Mean AUC ROC: {mean_auc_roc:.2f}")
        return mean_auc_roc, mean_acc, mean_precision, mean_recall

class XAIFG_Node(Explainer):
    def __init__(self, args):
        super(XAIFG_Node, self).__init__()

        self.args = args
        if args.dataset == 'Mutagenicity':
            self.data = MutagDataset(args=args)
        else:
            self.data = SyntheticDataset(args=args)
        self.model = self._prepare_model()
        if self.args.gnn_task == 'node':
            if self.args.gnn_type =='supervised':
                embeds = self.model.embedding((self.data.features_tensor, self.data.support_tensor)).cpu().detach().numpy()
            else:
                dummy_embeds = np.zeros(self.data.features.shape[0])
                embeds = dummy_embeds
            self.data.prepare_inductive(embeds=embeds)
            
        if args.gnn_type=='supervised':
            self.criterion = nn.CrossEntropyLoss()
        elif args.gnn_type=='unsupervised':
            self.criterion = nn.BCEWithLogitsLoss()
        self._cuda()

        
        # For validtest
        self.node_scores_vars = []
        self.node_scores = []

    def _cuda(self):
        """
        model
        dataset
        """
        device = self.args.device
        self.model = self.model.to(device)
        self.data.cuda()

    @torch.no_grad()
    def _test_model(self):
        """Check if the model works properly in the inductive setting."""
        model = self.model
        data = self.data
        model.eval()
        preds = []
        reals = []
        pred_scores = []
        target_nodeid = 0
        for node in data.nodes:
            output = model((data.sub_features[data.remap[node]], data.sub_support_tensors[data.remap[node]]))
            pred = torch.argmax(output, dim=1)
            real = np.argmax(data.sub_labels[data.remap[node]])

            pred_scores.append(output[:,1][target_nodeid].cpu())
            preds.append(pred[target_nodeid].cpu())
            reals.append(real)
        
        acc = accuracy_score(reals, preds)
        try:
            roc_auc = roc_auc_score(reals, pred_scores)
        except Exception as e:
            roc_auc = 0 
        print(acc, roc_auc)
        return acc, roc_auc
    
    def _prepare_model(self):
        data = self.data
        model_weight_path = self.args.model_weight_path
        
        # 개구린데..
        if self.args.gnn_task == 'node':
            if self.args.gnn_type == 'supervised':
                from models import GCN_Node as GCN
                model = GCN(args=self.args, input_dim=data.features.shape[1], output_dim=data.y_train.shape[1])
                model.load_state_dict(torch.load(model_weight_path, weights_only=True))
            if self.args.gnn_type == 'unsupervised':
                from torch_geometric.nn import GCN
                model = GCN(data.features.shape[1], hidden_channels=self.args.hidden_dim, num_layers=3)
                # model = GCN(data.features.shape[1], hidden_channels=self.args.hidden_dim, num_layers=5)
                model.load_state_dict(torch.load(model_weight_path, weights_only=True))

        elif self.args.gnn_task == 'graph':
            from models import GCN_Graph as GCN
            model = GCN(args=self.args, input_dim=data.feas.shape[-1], output_dim=data.labels.shape[1])
            model.load_state_dict(torch.load(model_weight_path, weights_only=True))

        model.eval()
        return model

    def _find_k_hops(self, adj, nodes, hops):
        indices = adj.coalesce().indices()
        current_level = set(nodes)
        visited = set(nodes)
        next_level = set()

        for _ in range(hops):
            for n in current_level:
                neighbors = indices[1][indices[0] == n].tolist()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
            
            current_level = next_level
            next_level = set()

            if not current_level:
                break
        
        return list(visited)

    def _hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem) # 이게 원래꺼
            # element_product += torch.mean(grad_elem * v_elem)
        
        return_grads = grad(element_product,model_params,create_graph=True)
        return return_grads
        
    def _gif_approxi(self, grads, nodeid, losses):
        iteration = self.args.iter
        scale = self.args.scale
        damp = 0
        grad_all, grad1, grad2 = grads
        loss, loss1, loss2 = losses
        original_params = [p.data for p in self.model.parameters() if p.requires_grad]
        v = tuple(grad1_t - grad2_t for grad1_t, grad2_t in zip(grad1,grad2))
        h_estimate = tuple(grad1_t - grad2_t for grad1_t, grad2_t in zip(grad1,grad2))
        for _ in range(iteration):
            model_params  = [p for p in self.model.parameters() if p.requires_grad]
            hv            = self._hvps(grad_all, model_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]
        
        params_change = [h_est * v1 for v1, h_est in zip(v, h_estimate)]

        # print(original_params)
        score = torch.mean(torch.stack([torch.mean(torch.abs(t)/(torch.abs(w)+0.000001)) for w, t in zip(original_params, params_change)]))
        # score = torch.mean(torch.stack([torch.mean(torch.abs(t)/(torch.abs(w))) for w, t in zip(original_params, params_change)]))
        # print(score)
        return score

    def _node_to_edge_scores(self, node_scores, adj):
        """_summary_

        Args:
            node_scores (List): index = node_id
        
        Returns:

        """

        task = self.args.task
        num_nodes = len(node_scores)
        if task == 'pos':
            # 연결된 간선만 포함
            # EDIT
            edge_scores = [
                (node_scores[i] * node_scores[j], (i, j))
                for i in range(num_nodes)
                for j in range(i + 1, num_nodes)
                if adj[i][j]
            ]
        elif task == 'neg':
            # 연결되지 않은 간선만 포함
            if self.args.gnn_type == 'unsupervised':
                edge_scores = [
                    # (node_scores[i] * node_scores[j], (i, j))
                    (abs(node_scores[i] - node_scores[j]), (i, j)) # for unsupervised(homophily)
                    for i in range(num_nodes)
                    for j in range(i + 1, num_nodes)
                    if not adj[i][j]
                ]
            elif self.args.gnn_type == 'supervised':
                eps = 1e-6
                edge_scores = [
                    # (1e+25 if node_scores[i] == node_scores[j] else node_scores[i] * node_scores[j], (i, j)) # mode2
                    # ([node_scores[i]/(node_scores[j]+eps) + node_scores[j]/(node_scores[i]+eps)], (i, j)) # 내생각
                    # ([1/(node_scores[i]+eps) + 1/(node_scores[j]+eps)], (i, j)) # 내생각
                    (node_scores[i] * node_scores[j], (i, j))
                    for i in range(num_nodes)
                    for j in range(i + 1, num_nodes)
                    if not adj[i][j]
                ]
        
        # print(edge_scores)
        return edge_scores
        
    def _pos_singlegraph(self, nodeid):
        data = self.data
        model = self.model
        dataset_name = self.args.dataset
        node_scores = []
        score_edges = []
        # 함
        if self.args.gnn_task=='node' and self.args.gnn_type=='supervised':
            original_feature_tensor = data.sub_features[data.remap[nodeid]]
            original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
            label = data.sub_label_tensors[data.remap[nodeid]]
            indices = original_support_tensor.coalesce().indices()
            values = original_support_tensor.coalesce().values()
            size = original_support_tensor.coalesce().size()
        
            influenced_hops = len([int(s) for s in self.args.hiddens.split('-')])
            
            num_nodes = size[0]
            for node in range(num_nodes):
                mask = ~((indices[0] == node) | (indices[1] == node)) # 노드기준
                new_indices = indices[:, mask]
                new_values = values[mask]

                perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
                original_output = model((original_feature_tensor, original_support_tensor), training=False)
                perturbed_output = model((original_feature_tensor, perturbed_support_tensor), training=False)


                influenced_nodes = self._find_k_hops(original_support_tensor, [node], influenced_hops)
                # influenced_nodes = self._find_k_hops(perturbed_support_tensor, [0], influenced_hops)
                # influenced_nodes = [0]

                mask1 = np.array([False] * perturbed_output.shape[0])
                mask1[influenced_nodes] = True
                
                mask2 = mask1

                original_whole_output = model((data.features_tensor, data.support_tensor))
                all_label_tensor = torch.tensor(data.all_label, dtype=torch.float32, device=self.args.device)

                loss = self.criterion(original_whole_output[data.train_mask_tensor], all_label_tensor[data.train_mask_tensor])
                loss1 = self.criterion(original_output[mask1], label[mask1])
                loss2 = self.criterion(perturbed_output[mask2], label[mask2])
                model_params = [p for p in model.parameters() if p.requires_grad]
                
                grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                node_scores.append(score.item()) # index = node번호


            # normalized_node_scores = (node_scores - np.min(node_scores)) / (np.max(node_scores) - np.min(node_scores))
            # self.node_scores.extend(normalized_node_scores)
            # self.node_scores_var = np.var(normalized_node_scores) # for validtest
            

            score_edges = self._node_to_edge_scores(node_scores, original_support_tensor.to_dense())
            score_edges.sort(reverse=True, key=lambda x: x[0])
        
        # 함
        elif self.args.gnn_task=='node' and self.args.gnn_type=='unsupervised':
            original_sub_feature_tensor = data.sub_features[data.remap[nodeid]]
            original_sub_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
            indices = original_sub_support_tensor.coalesce().indices()
            values = original_sub_support_tensor.coalesce().values()
            size = original_sub_support_tensor.coalesce().size()
            num_node = size[0]

            original_sub_edge_index = original_sub_support_tensor.to_dense().nonzero(as_tuple=False).t()

            ## label
            row = torch.arange(num_node).repeat(num_node)
            col = torch.arange(num_node).unsqueeze(1).repeat(1, num_node).view(-1)
            sub_edge_index_all = torch.stack([row, col], dim=0) 
            
            # whole_num_node = data.support_tensor.coalesce().size()[0]
            # whole_row = torch.arange(whole_num_node).repeat(whole_num_node)
            # whole_col = torch.arange(whole_num_node).unsqueeze(1).repeat(1, whole_num_node).view(-1)
            # whole_edge_index_all = torch.stack([whole_row, whole_col], dim=0).to(self.args.device)
            # edge_index_whole = data.support_tensor.coalesce().indices()
            
            sub_edge_label_tensor = torch.zeros(sub_edge_index_all.size(1), dtype=torch.float32, device=self.args.device)
            src = original_sub_edge_index[0]
            dst = original_sub_edge_index[1]
            all_indices = src * num_node + dst
            sub_edge_label_tensor[all_indices] = 1

            # whole_edge_index = data.support_tensor.to_dense().nonzero(as_tuple=False).t()
            # full_edge_label_tensor = torch.zeros(whole_num_node*whole_num_node, dtype=torch.float32, device=self.args.device)
            # src = whole_edge_index[0]
            # dst = whole_edge_index[1]
            # all_indices = src * whole_num_node + dst
            # full_edge_label_tensor[all_indices] = 1

            for node in range(num_node):
                mask = ~((indices[0] == node) | (indices[1] == node))
                new_indices = indices[:, mask]
                new_values = values[mask]
                
                perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
                perturbed_edge_index = perturbed_support_tensor.to_dense().nonzero(as_tuple=False).t()

                # whole
                # h_whole = model(data.features_tensor, edge_index_whole)
                # h_whole_src = h_whole[whole_edge_index_all[0]]
                # h_whole_dst = h_whole[whole_edge_index_all[1]]
                # pred_whole = (h_whole_src * h_whole_dst).sum(dim=-1)
                
                # original
                h_ori = model(original_sub_feature_tensor, original_sub_edge_index)
                h_ori_src = h_ori[sub_edge_index_all[0]]
                h_ori_dst = h_ori[sub_edge_index_all[1]]
                pred_ori = (h_ori_src * h_ori_dst).sum(dim=-1)
                
                # perturb
                h_per = model(original_sub_feature_tensor, perturbed_edge_index)
                h_per_src = h_per[sub_edge_index_all[0]]
                h_per_dst = h_per[sub_edge_index_all[1]]
                pred_per = (h_per_src * h_per_dst).sum(dim=-1)
                
                # loss = self.criterion(pred_whole, full_edge_label_tensor)
                loss = self.criterion(pred_ori, sub_edge_label_tensor)
                loss1 = self.criterion(pred_ori, sub_edge_label_tensor)
                loss2 = self.criterion(pred_per, sub_edge_label_tensor)
                
                model_params = [p for p in model.parameters() if p.requires_grad]
                
                grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                node_scores.append(score.item()) # index = node번호
                
            # normalized_node_scores = (node_scores - np.min(node_scores)) / (np.max(node_scores) - np.min(node_scores))
            # self.node_scores.extend(normalized_node_scores)
            # # self.node_scores.extend(node_scores)
            # self.node_scores_var = np.var(normalized_node_scores) # for validtest
            # # self.node_scores_var = np.var(node_scores) # for validtest

            score_edges = self._node_to_edge_scores(node_scores, original_sub_support_tensor.to_dense())
            score_edges.sort(reverse=True, key=lambda x: x[0])

        # 함
        elif self.args.gnn_task=='graph' and self.args.gnn_type=='supervised':
            if dataset_name == 'BA-2motif' or dataset_name == 'Mutagenicity':
                adj, fea, label = data.adjs[nodeid], data.feas[nodeid], data.labels[nodeid]
                graph = adj, fea, label
                label = torch.tensor(label)
                fea_tensor = torch.tensor(fea, dtype=torch.float32).unsqueeze(0)
                label_tensor = torch.argmax(label).unsqueeze(0).to(args.device)
                num_nodes = adj.shape[-1]

                for node in range(num_nodes):
                    new_adj = copy.deepcopy(adj)
                    
                    for adj_node in range(num_nodes):
                        new_adj[node][adj_node] = False
                        new_adj[adj_node][node] = False

                    original_output = model((fea_tensor, torch.tensor(adj, dtype=torch.float32).unsqueeze(0)))
                    perturbed_output = model((fea_tensor, torch.tensor(new_adj, dtype=torch.float32).unsqueeze(0)))

                    loss = cross_entropy(original_output, label_tensor)
                    loss1 = cross_entropy(original_output, label_tensor)
                    loss2 = cross_entropy(perturbed_output, label_tensor)

                    model_params = [p for p in model.parameters() if p.requires_grad]
                    
                    grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                    grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                    grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                    score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2)) # node에 대한 score
                    node_scores.append(score.item()) # index = node번호

                # for visualization
                self.node_scores = node_scores

                score_edges = self._node_to_edge_scores(node_scores, adj)
                score_edges.sort(reverse=True, key=lambda x: x[0])

        # 원래없고        
        elif self.args.gnn_task=='graph' and self.args.gnn_type=='unsupervised':
            """
            Not implemented yet (no appropriate unsupervised setting for graph classification).
            """


        normalized_node_scores = (node_scores - np.min(node_scores)) / (np.max(node_scores) - np.min(node_scores))
        self.node_scores.extend(normalized_node_scores)
        self.node_scores_var = np.var(normalized_node_scores) # for validtest

        # For Visualization
        self.node_scores_singlegraph = node_scores
        
        return score_edges

    def _neg_singlegraph(self, nodeid):
        data = self.data
        model = self.model
        dataset_name = self.args.dataset
        node_scores = []
        score_edges = []

        if self.args.gnn_task=='node' and self.args.gnn_type=='supervised':
            if dataset_name == 'syn3':
                original_feature_tensor = data.sub_features[data.remap[nodeid]]
                original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
                label = data.sub_label_tensors[data.remap[nodeid]]
                indices = original_support_tensor.coalesce().indices()
                values = original_support_tensor.coalesce().values()
                size = original_support_tensor.coalesce().size()
            
                influenced_hops = len([int(s) for s in self.args.hiddens.split('-')])
                
                num_nodes = size[0]

                # node와 연결된 모든 node 끊
                for node in range(num_nodes):
                    mask = ~((indices[0] == node) | (indices[1] == node))
                    new_indices = indices[:, mask]
                    new_values = values[mask]

                # node랑 연결되지 않은 모든 node들 연결
                # for node in range(num_nodes):
                #     new_indices = indices.clone()
                #     new_values = values.clone()

                #     connected_from = indices[1][indices[0] == node]
                #     connected_to   = indices[0][indices[1] == node]
                #     connected_nodes = torch.cat([connected_from, connected_to]).unique()

                #     all_nodes = torch.arange(num_nodes, device=indices.device)
                #     exclude = torch.cat([connected_nodes, torch.tensor([node], device=indices.device)])
                #     unconnected_nodes = all_nodes[~torch.isin(all_nodes, exclude)]

                #     if unconnected_nodes.numel() > 0:
                #         new_edges = torch.stack([
                #             torch.full((len(unconnected_nodes),), node, dtype=torch.long, device=indices.device),
                #             unconnected_nodes
                #         ], dim=0)

                #         new_edge_values = torch.ones(len(unconnected_nodes), device=indices.device)

                #         new_indices = torch.cat([new_indices, new_edges], dim=1)
                #         new_values = torch.cat([new_values, new_edge_values], dim=0)
                    
                    
                    perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
                    original_output = model((original_feature_tensor, original_support_tensor), training=False)
                    perturbed_output = model((original_feature_tensor, perturbed_support_tensor), training=False)

                    influenced_nodes = self._find_k_hops(original_support_tensor, [node], influenced_hops)
                    
                    mask1 = np.array([False] * perturbed_output.shape[0])
                    # mask1[influenced_nodes] = True
                    mask1[0] = True

                    mask2 = mask1

                    original_whole_output = model((data.features_tensor, data.support_tensor))
                    all_label_tensor = torch.tensor(data.all_label, dtype=torch.float32, device=self.args.device)

                    loss = self.criterion(original_whole_output[data.train_mask_tensor], all_label_tensor[data.train_mask_tensor])
                    loss1 = self.criterion(original_output[mask1], label[mask1])
                    loss2 = self.criterion(perturbed_output[mask2], label[mask2])
                    model_params = [p for p in model.parameters() if p.requires_grad]
                    
                    grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                    grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                    grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                    score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                    node_scores.append(score.item()) # index = node번호

                normalized_node_scores = (node_scores - np.min(node_scores)) / (np.max(node_scores) - np.min(node_scores))
                
                # self.node_scores.extend(normalized_node_scores)
                # self.node_scores_var = np.var(normalized_node_scores) # for validtest
                # self.cur_node_scores = normalized_node_scores

                score_edges = self._node_to_edge_scores(node_scores, original_support_tensor.to_dense())
                score_edges.sort(reverse=True, key=lambda x: x[0])
                
            elif dataset_name == 'syn4':
                """"""
                original_feature_tensor = data.sub_features[data.remap[nodeid]]
                original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
                label = data.sub_label_tensors[data.remap[nodeid]]
                # label = torch.tensor(data.sub_labels[data.remap[nodeid]])
                indices = original_support_tensor.coalesce().indices()
                values = original_support_tensor.coalesce().values()
                size = original_support_tensor.coalesce().size()
                
                influenced_hops = len([int(s) for s in self.args.hiddens.split('-')])
                
                deletion_candidates = []
                for edge_index in range(len(values)):
                    src, des = indices[0][edge_index].item(), indices[1][edge_index].item()
                    if src >= des:
                        continue
                    if data.sub_edge_labels[data.remap[nodeid]].todense()[src,des] or data.sub_edge_labels[data.remap[nodeid]].todense()[des,src]:
                        deletion_candidates.append((src, des))
                    
                # edges_to_delete = random.sample(deletion_candidates, k)
                edges_to_delete = random.sample(deletion_candidates, 1) # 일단 k=1 고정

                for src_del, des_del in edges_to_delete:
                    del_mask = ~(((indices[0] == src_del) & (indices[1] == des_del)) | ((indices[0] == des_del) & (indices[1] == src_del)))
                    del_indices = indices[:, del_mask]
                    del_values = values[del_mask]
                    del_original_support_tensor = torch.sparse_coo_tensor(del_indices, del_values, size)

                    num_nodes = size[0]
                    # node와 연결된 모든 node 끊
                    # for node in range(num_nodes):
                    #     mask = ~((del_indices[0] == node) | (del_indices[1] == node))
                    #     new_indices = del_indices[:, mask]
                    #     new_values = del_values[mask]

                    # node랑 연결되지 않은 모든 node들 연결
                    for node in range(num_nodes):
                        new_indices = indices.clone()
                        new_values = values.clone()

                        connected_from = indices[1][indices[0] == node]
                        connected_to   = indices[0][indices[1] == node]
                        connected_nodes = torch.cat([connected_from, connected_to]).unique()

                        all_nodes = torch.arange(num_nodes, device=indices.device)
                        exclude = torch.cat([connected_nodes, torch.tensor([node], device=indices.device)])
                        unconnected_nodes = all_nodes[~torch.isin(all_nodes, exclude)]

                        if unconnected_nodes.numel() > 0:
                            new_edges = torch.stack([
                                torch.full((len(unconnected_nodes),), node, dtype=torch.long, device=indices.device),
                                unconnected_nodes
                            ], dim=0)

                            new_edge_values = torch.ones(len(unconnected_nodes), device=indices.device)

                            new_indices = torch.cat([new_indices, new_edges], dim=1)
                            new_values = torch.cat([new_values, new_edge_values], dim=0)
                    
                        perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
                        original_output = model((original_feature_tensor, del_original_support_tensor), training=False)
                        perturbed_output = model((original_feature_tensor, perturbed_support_tensor), training=False)

                        # influenced_nodes = self._find_k_hops(perturbed_support_tensor, [node], influenced_hops)
                        influenced_nodes = self._find_k_hops(del_original_support_tensor, [node], influenced_hops)

                        mask1 = np.array([False] * perturbed_output.shape[0])
                        mask1[influenced_nodes] = True
                        
                        mask2 = mask1

                        original_whole_output = model((data.features_tensor, data.support_tensor))
                        all_label_tensor = torch.tensor(data.all_label, dtype=torch.float32, device=self.args.device)

                        loss = self.criterion(original_whole_output[data.train_mask_tensor], all_label_tensor[data.train_mask_tensor])
                        loss1 = self.criterion(original_output[mask1], label[mask1])
                        loss2 = self.criterion(perturbed_output[mask2], label[mask2])
                        model_params = [p for p in model.parameters() if p.requires_grad]
                        
                        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                        score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                        node_scores.append(score.item()) # index = node번호


                # normalized_node_scores = (node_scores - np.min(node_scores)) / (np.max(node_scores) - np.min(node_scores))
                # self.node_scores.extend(normalized_node_scores)
                # self.node_scores_var = np.var(normalized_node_scores) # for validtest

                score_edges = self._node_to_edge_scores(node_scores, del_original_support_tensor.to_dense())
                # score_edges.sort(reverse=True, key=lambda x: x[0])
                
        elif self.args.gnn_task=='node' and self.args.gnn_type=='unsupervised':
            original_sub_feature_tensor = data.sub_features[data.remap[nodeid]]
            original_sub_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
            indices = original_sub_support_tensor.coalesce().indices()
            values = original_sub_support_tensor.coalesce().values()
            size = original_sub_support_tensor.coalesce().size()
            num_node = size[0]

            original_sub_edge_index = original_sub_support_tensor.to_dense().nonzero(as_tuple=False).t()

            ## label
            row = torch.arange(num_node).repeat(num_node)
            col = torch.arange(num_node).unsqueeze(1).repeat(1, num_node).view(-1)
            sub_edge_index_all = torch.stack([row, col], dim=0) 
            
            # whole_num_node = data.support_tensor.coalesce().size()[0]
            # whole_row = torch.arange(whole_num_node).repeat(whole_num_node)
            # whole_col = torch.arange(whole_num_node).unsqueeze(1).repeat(1, whole_num_node).view(-1)
            # whole_edge_index_all = torch.stack([whole_row, whole_col], dim=0).to(self.args.device)
            # edge_index_whole = data.support_tensor.coalesce().indices()
            
            sub_edge_label_tensor = torch.zeros(sub_edge_index_all.size(1), dtype=torch.float32, device=self.args.device)
            src = original_sub_edge_index[0]
            dst = original_sub_edge_index[1]
            all_indices = src * num_node + dst
            sub_edge_label_tensor[all_indices] = 1

            # whole_edge_index = data.support_tensor.to_dense().nonzero(as_tuple=False).t()
            # full_edge_label_tensor = torch.zeros(whole_num_node*whole_num_node, dtype=torch.float32, device=self.args.device)
            # src = whole_edge_index[0]
            # dst = whole_edge_index[1]
            # all_indices = src * whole_num_node + dst
            # full_edge_label_tensor[all_indices] = 1

            # nr
            for node in range(num_node):
                mask = ~((indices[0] == node) | (indices[1] == node))
                new_indices = indices[:, mask]
                new_values = values[mask]
                
            # nc
            # for node in range(num_node):
            #     new_indices = indices.clone()
            #     new_values = values.clone()

            #     connected_from = indices[1][indices[0] == node]
            #     connected_to   = indices[0][indices[1] == node]
            #     connected_nodes = torch.cat([connected_from, connected_to]).unique()

            #     all_nodes = torch.arange(num_node, device=indices.device)
            #     exclude = torch.cat([connected_nodes, torch.tensor([node], device=indices.device)])
            #     unconnected_nodes = all_nodes[~torch.isin(all_nodes, exclude)]

            #     if unconnected_nodes.numel() > 0:
            #         new_edges = torch.stack([
            #             torch.full((len(unconnected_nodes),), node, dtype=torch.long, device=indices.device),
            #             unconnected_nodes
            #         ], dim=0)

            #         new_edge_values = torch.ones(len(unconnected_nodes), device=indices.device)

            #         new_indices = torch.cat([new_indices, new_edges], dim=1)
            #         new_values = torch.cat([new_values, new_edge_values], dim=0)
                
                
                perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
                perturbed_edge_index = perturbed_support_tensor.to_dense().nonzero(as_tuple=False).t()

                # whole
                # h_whole = model(data.features_tensor, edge_index_whole)
                # h_whole_src = h_whole[whole_edge_index_all[0]]
                # h_whole_dst = h_whole[whole_edge_index_all[1]]
                # pred_whole = (h_whole_src * h_whole_dst).sum(dim=-1)
                
                # original
                h_ori = model(original_sub_feature_tensor, original_sub_edge_index)
                h_ori_src = h_ori[sub_edge_index_all[0]]
                h_ori_dst = h_ori[sub_edge_index_all[1]]
                pred_ori = (h_ori_src * h_ori_dst).sum(dim=-1)
                
                # perturb
                h_per = model(original_sub_feature_tensor, perturbed_edge_index)
                h_per_src = h_per[sub_edge_index_all[0]]
                h_per_dst = h_per[sub_edge_index_all[1]]
                pred_per = (h_per_src * h_per_dst).sum(dim=-1)
                
                # loss = self.criterion(pred_whole, full_edge_label_tensor)
                loss = self.criterion(pred_ori, sub_edge_label_tensor)
                loss1 = self.criterion(pred_ori, sub_edge_label_tensor)
                loss2 = self.criterion(pred_per, sub_edge_label_tensor)
                
                model_params = [p for p in model.parameters() if p.requires_grad]
                
                grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
                node_scores.append(score.item()) # index = node번호
            
            # normalized_node_scores = (node_scores - np.min(node_scores)) / (np.max(node_scores) - np.min(node_scores))
            # self.node_scores.extend(normalized_node_scores)
            # self.node_scores_var = np.var(normalized_node_scores) # for validtest

            score_edges = self._node_to_edge_scores(node_scores, original_sub_support_tensor.to_dense())

        elif self.args.gnn_task=='graph' and self.args.gnn_type=='supervised':
            if dataset_name == 'BA-2motif':
                adj, fea, label = data.adjs[nodeid], data.feas[nodeid], data.labels[nodeid]
                graph = adj, fea, label
                label = torch.tensor(label)
                fea_tensor = torch.tensor(fea, dtype=torch.float32).unsqueeze(0)
                label_tensor = torch.argmax(label).unsqueeze(0).to(args.device)
                num_nodes = adj.shape[-1]

                # node_scores = [0 for _ in range(num_nodes)]
                node_scores = []
                for node in range(num_nodes):
                    new_adj = copy.deepcopy(adj)
                    
                    # nr
                    for adj_node in range(num_nodes):
                        new_adj[node][adj_node] = False
                        new_adj[adj_node][node] = False

                    # nc
                    # for adj_node in range(num_nodes):
                    #     new_adj[node][adj_node] = True
                    #     new_adj[adj_node][node] = True
                
                    original_output = model((fea_tensor, torch.tensor(adj, dtype=torch.float32).unsqueeze(0)))
                    perturbed_output = model((fea_tensor, torch.tensor(new_adj, dtype=torch.float32).unsqueeze(0)))

                    loss = cross_entropy(original_output, label_tensor)
                    loss1 = cross_entropy(original_output, label_tensor)    
                    loss2 = cross_entropy(perturbed_output, label_tensor)

                    model_params = [p for p in model.parameters() if p.requires_grad]
                    
                    grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                    grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                    grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                    score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2)) # node에 대한 score
                    node_scores.append(score.item()) # index = node번호


                normalized_node_scores = (node_scores - np.min(node_scores)) / (np.max(node_scores) - np.min(node_scores))
                self.node_scores.extend(normalized_node_scores)
                self.node_scores_var = np.var(normalized_node_scores) # for validtest

                score_edges = self._node_to_edge_scores(node_scores, adj)

        elif self.args.gnn_task=='graph' and self.args.gnn_type=='unsupervised':
            """
            Not implemented yet (no appropriate unsupervised setting for graph classification).
            """

            
        # if dataset_name == 'syn3':
        #     original_feature_tensor = data.sub_features[data.remap[nodeid]]
        #     original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
        #     label = data.sub_label_tensors[data.remap[nodeid]]
        #     indices = original_support_tensor.coalesce().indices()
        #     values = original_support_tensor.coalesce().values()
        #     size = original_support_tensor.coalesce().size()
        
        #     influenced_hops = len([int(s) for s in self.args.hiddens.split('-')])
            
        #     num_nodes = size[0]
        #     for node in range(num_nodes):
        #         mask = ~((indices[0] == node) | (indices[1] == node))
        #         new_indices = indices[:, mask]
        #         new_values = values[mask]

        #         perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
        #         original_output = model((original_feature_tensor, original_support_tensor), training=False)
        #         perturbed_output = model((original_feature_tensor, perturbed_support_tensor), training=False)

        #         influenced_nodes = self._find_k_hops(perturbed_support_tensor, [node], influenced_hops)

        #         mask1 = np.array([False] * perturbed_output.shape[0])
        #         mask1[influenced_nodes] = True
                
        #         mask2 = mask1

        #         original_whole_output = model((data.features_tensor, data.support_tensor))
        #         all_label_tensor = torch.tensor(data.all_label, dtype=torch.float32, device=self.args.device)

        #         loss = self.criterion(original_whole_output[data.train_mask_tensor], all_label_tensor[data.train_mask_tensor])
        #         loss1 = self.criterion(original_output[mask1], label[mask1])
        #         loss2 = self.criterion(perturbed_output[mask2], label[mask2])
        #         model_params = [p for p in model.parameters() if p.requires_grad]
                
        #         grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        #         grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        #         grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

        #         score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
        #         node_scores.append(score.item()) # index = node번호

        #     score_edges = self._node_to_edge_scores(node_scores, original_support_tensor.to_dense())
        #     score_edges.sort(reverse=True, key=lambda x: x[0])
            
        # elif dataset_name == 'syn4':
        #     """"""
        #     original_feature_tensor = data.sub_features[data.remap[nodeid]]
        #     original_support_tensor = data.sub_support_tensors[data.remap[nodeid]]
        #     label = data.sub_label_tensors[data.remap[nodeid]]
        #     # label = torch.tensor(data.sub_labels[data.remap[nodeid]])
        #     indices = original_support_tensor.coalesce().indices()
        #     values = original_support_tensor.coalesce().values()
        #     size = original_support_tensor.coalesce().size()
            
        #     influenced_hops = len([int(s) for s in self.args.hiddens.split('-')])
            
        #     deletion_candidates = []
        #     for edge_index in range(len(values)):
        #         src, des = indices[0][edge_index].item(), indices[1][edge_index].item()
        #         if src >= des:
        #             continue
        #         if data.sub_edge_labels[data.remap[nodeid]].todense()[src,des] or data.sub_edge_labels[data.remap[nodeid]].todense()[des,src]:
        #             deletion_candidates.append((src, des))
                
        #     # edges_to_delete = random.sample(deletion_candidates, k)
        #     edges_to_delete = random.sample(deletion_candidates, 1) # 일단 k=1 고정

        #     for src_del, des_del in edges_to_delete:
        #         del_mask = ~(((indices[0] == src_del) & (indices[1] == des_del)) | ((indices[0] == des_del) & (indices[1] == src_del)))
        #         del_indices = indices[:, del_mask]
        #         del_values = values[del_mask]
        #         del_original_support_tensor = torch.sparse_coo_tensor(del_indices, del_values, size)

        #         num_nodes = size[0]
        #         for node in range(num_nodes):
        #             mask = ~((del_indices[0] == node) | (del_indices[1] == node))
        #             new_indices = del_indices[:, mask]
        #             new_values = del_values[mask]
                    
        #             perturbed_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)
        #             original_output = model((original_feature_tensor, del_original_support_tensor), training=False)
        #             perturbed_output = model((original_feature_tensor, perturbed_support_tensor), training=False)

        #             influenced_nodes = self._find_k_hops(perturbed_support_tensor, [node], influenced_hops)

        #             mask1 = np.array([False] * perturbed_output.shape[0])
        #             mask1[influenced_nodes] = True
                    
        #             mask2 = mask1

        #             original_whole_output = model((data.features_tensor, data.support_tensor))
        #             all_label_tensor = torch.tensor(data.all_label, dtype=torch.float32, device=self.args.device)

        #             loss = self.criterion(original_whole_output[data.train_mask_tensor], all_label_tensor[data.train_mask_tensor])
        #             loss1 = self.criterion(original_output[mask1], label[mask1])
        #             loss2 = self.criterion(perturbed_output[mask2], label[mask2])
        #             model_params = [p for p in model.parameters() if p.requires_grad]
                    
        #             grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        #             grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        #             grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

        #             score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2))
        #             node_scores.append(score.item()) # index = node번호

        #     score_edges = self._node_to_edge_scores(node_scores, del_original_support_tensor.to_dense())
        #     score_edges.sort(reverse=True, key=lambda x: x[0])
            
        # elif dataset_name == 'BA-2motif':
            adj, fea, label = data.adjs[nodeid], data.feas[nodeid], data.labels[nodeid]
            graph = adj, fea, label
            label = torch.tensor(label)
            fea_tensor = torch.tensor(fea, dtype=torch.float32).unsqueeze(0)
            label_tensor = torch.argmax(label).unsqueeze(0).to(args.device)
            num_nodes = adj.shape[-1]

            # node_scores = [0 for _ in range(num_nodes)]
            node_scores = []
            for node in range(num_nodes):
                new_adj = copy.deepcopy(adj)
                
                for adj_node in range(num_nodes):
                    new_adj[node][adj_node] = False
                    new_adj[adj_node][node] = False
            
                original_output = model((fea_tensor, torch.tensor(adj, dtype=torch.float32).unsqueeze(0)))
                perturbed_output = model((fea_tensor, torch.tensor(new_adj, dtype=torch.float32).unsqueeze(0)))

                loss = cross_entropy(original_output, label_tensor)
                loss1 = cross_entropy(original_output, label_tensor)    
                loss2 = cross_entropy(perturbed_output, label_tensor)

                model_params = [p for p in model.parameters() if p.requires_grad]
                
                grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
                grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
                grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

                score = self._gif_approxi((grad_all, grad1, grad2), nodeid, (loss, loss1, loss2)) # node에 대한 score
                node_scores.append(score.item()) # index = node번호

            score_edges = self._node_to_edge_scores(node_scores, adj)


        normalized_node_scores = (node_scores - np.min(node_scores)) / (np.max(node_scores) - np.min(node_scores))
        self.node_scores.extend(normalized_node_scores)
        self.node_scores_var = np.var(normalized_node_scores) # for validtest

        # For Visualization
        self.node_scores_singlegraph = node_scores

        score_edges.sort(reverse=True, key=lambda x: x[0])

        # For unsup fidelity test
        self.score_edges = score_edges

        return score_edges
    
    def edge_influences(self):
        auc_rocs = []
        acc_list = []
        precision_list = []
        recall_list = []

        allnodes = self.data.nodes
        for node in tqdm(allnodes):
            if self.args.task == 'neg':
                score_edges = self._neg_singlegraph(node)
                edge_reals, topk = self._get_neg_reals(node, score_edges, return_topk=True)
            elif self.args.task == 'pos':
                score_edges = self._pos_singlegraph(node)
                edge_reals, topk = self._get_pos_reals(node, score_edges, return_topk=True)
            
            all_reals = [1 if edge_reals[edge] else 0 for _, edge in score_edges]
            topk_scores = score_edges[:topk]
            topk_edges = {edge for _, edge in topk_scores}
            preds = [1 if edge in topk_edges else 0 for _, edge in score_edges]

            acc = accuracy_score(all_reals, preds)
            precision = precision_score(all_reals, preds, zero_division=0)
            recall = recall_score(all_reals, preds, zero_division=0)

            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            try:
                scores = [score for score, _ in score_edges]
                rocauc = roc_auc_score(all_reals, scores)
                auc_rocs.append(rocauc)
                self.node_scores_vars.append(self.node_scores_var)

                
                # print('-'*30)
                print(f'node {node:4}, rocauc {rocauc}')

                # if rocauc < 0.1 or rocauc > 0.8:
                # if 0.2 < rocauc and rocauc < 0.3:
                if False:
                    remapped_node = self.data.remap[node]
                    adj = self.data.sub_support_tensors[remapped_node].cpu().to_dense()

                    print('changed edges:')
                    for edge, real in edge_reals.items():
                        if real:
                            print(edge)
                    print('-'*30)

                    node_scores = list(self.cur_node_scores)
                    indexed_scores = sorted(enumerate(node_scores), key=lambda x: x[1], reverse=True)

                    for node_idx, node_score in indexed_scores:
                        print(f'{node_idx:4}, {node_score:8.4f}')
                    plot_adj(adj, show=True, save=False, graph_index=True)
                # print('-'*30)
                
                # for vis
                # if rocauc <= 0.02:
                # if rocauc == 0.0:
                if rocauc >= 0.9:
                    """"""
                    # sup node pos vis
                    """"""
                    # remapped_node = self.data.remap[node]
                    # adj = self.data.sub_support_tensors[remapped_node].cpu().to_dense()
                    # node_scores = self.node_scores_singlegraph
                    # normalized_node_scores = [(x - min(node_scores)) / (max(node_scores) - min(node_scores)) if max(node_scores) > min(node_scores) else 0 for x in node_scores]
                    # normalized_node_scores = [x * 10 for x in normalized_node_scores] # 0~10 norm
                    # # print(normalized_node_scores)
                    # topk = 12 # cycle=6, grid=12
                    # plot_pos_adj_both(
                    #     adj = adj,
                    #     node_scores = normalized_node_scores,
                    #     special_edges=score_edges,
                    #     topk=topk,
                    #     show=True,
                    #     save=False,
                    #     pic_name='',
                    #     with_labels=False
                    # )
                    
                    # sup node neg vis
                    """"""
                    # remapped_node = self.data.remap[node]
                    # adj = self.data.sub_support_tensors[remapped_node].cpu().to_dense()
                    # node_scores = self.node_scores_singlegraph
                    # normalized_node_scores = [(x - min(node_scores)) / (max(node_scores) - min(node_scores)) if max(node_scores) > min(node_scores) else 0 for x in node_scores]
                    # normalized_node_scores = [x * 10 for x in normalized_node_scores] # 0~10 norm
                    # topk=1
                    # plot_neg_adj_both(
                    #     adj = adj,
                    #     node_scores = normalized_node_scores,
                    #     special_edges=score_edges,
                    #     topk=topk,
                    #     show=True,
                    #     save=False,
                    #     pic_name=''
                    # )

                    

                    # sup graph pos vis
                    """"""
                    # adj = self.data.adjs[node]
                    # node_scores = self.node_scores_singlegraph
                    # normalized_node_scores = [(x - min(node_scores)) / (max(node_scores) - min(node_scores)) if max(node_scores) > min(node_scores) else 0 for x in node_scores]
                    # normalized_node_scores = [x * 10 for x in normalized_node_scores] # 0~10 norm
                    
                    # # plot_adj(adj)
                    # node_to_show = [20,21,22,23,24, 0,5,7,17, 10,14] # 너무 커서 얘네만 살리기
                    # for node_ in range(25):
                    #     if node_ not in node_to_show:
                    #         for i in range(25):
                    #             adj[node_][i]=False
                    #             adj[i][node_]=False
                    # topk = 6 # label0=5, label1=6
                    # plot_pos_adj_both(
                    #     adj = adj,
                    #     node_scores = normalized_node_scores,
                    #     special_edges=score_edges,
                    #     topk=topk,
                    #     show=True,
                    #     save=False,
                    #     pic_name='',
                    #     with_labels=False
                    # )
                    

                    # sup graph neg vis
                    """"""
                    # if node == 25 or node == 80:
                    #     continue
                    # adj = self.data.adjs[node]
                    # node_scores = self.node_scores_singlegraph
                    # normalized_node_scores = [(x - min(node_scores)) / (max(node_scores) - min(node_scores)) if max(node_scores) > min(node_scores) else 0 for x in node_scores]
                    # normalized_node_scores = [x * 10 for x in normalized_node_scores] # 0~10 norm
                    # # plot_adj(adj)
                    # node_to_show = [20,21,22,23,24, 0,2,1, 3,9] # 너무 커서 얘네만 살리기
                    # for node_ in range(25):
                    #     if node_ not in node_to_show:
                    #         for i in range(25):
                    #             adj[node_][i]=False
                    #             adj[i][node_]=False
                    # topk=1
                    # plot_neg_adj_both(
                    #     adj = adj,
                    #     node_scores = normalized_node_scores,
                    #     special_edges=score_edges,
                    #     topk=topk,
                    #     show=True,
                    #     save=False,
                    #     pic_name=''
                    # )


                    # unsup node pos vis
                    """"""
                    # remapped_node = self.data.remap[node]
                    # adj = self.data.sub_support_tensors[remapped_node].cpu().to_dense()
                    # node_scores = self.node_scores_singlegraph
                    # normalized_node_scores = [(x - min(node_scores)) / (max(node_scores) - min(node_scores)) if max(node_scores) > min(node_scores) else 0 for x in node_scores]
                    # normalized_node_scores = [x * 30 for x in normalized_node_scores] # 0~10 norm
                    # score_edges.reverse()
                    # if len(node_scores) >= 20:
                    #     continue
                    # # node만
                    # # plot_pos_adj_node(
                    # #     adj = adj,
                    # #     node_scores = normalized_node_scores,
                    # #     show=True,
                    # #     save=False,
                    # #     pic_name='',
                    # # )
                    # # 둘다
                    # topk = 6 # cycle=6, grid=12
                    # plot_pos_adj_both(
                    #     adj = adj,
                    #     node_scores = normalized_node_scores,
                    #     special_edges=score_edges,
                    #     topk=topk,
                    #     show=True,
                    #     save=False,
                    #     pic_name='',
                    #     with_labels=False
                    # )
                    
                    

                    # unsup node neg vis
                    # remapped_node = self.data.remap[node]
                    # adj = self.data.sub_support_tensors[remapped_node].cpu().to_dense()
                    # node_scores = self.node_scores_singlegraph
                    # normalized_node_scores = [(x - min(node_scores)) / (max(node_scores) - min(node_scores)) if max(node_scores) > min(node_scores) else 0 for x in node_scores]
                    # normalized_node_scores = [x * 10 for x in normalized_node_scores] # 0~10 norm
                    # topk=2
                    # plot_neg_adj_both(
                    #     adj = adj,
                    #     node_scores = normalized_node_scores,
                    #     special_edges=score_edges,
                    #     topk=topk,
                    #     show=True,
                    #     save=False,
                    #     pic_name=''
                    # )

                # Graph classification Visualization
                # if rocauc >= 0.9:
            except Exception as e:
                print(e)
                # return float('nan'),float('nan'),float('nan'),float('nan') # for validtest
                continue

        mean_acc = np.mean(acc_list)
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_auc_roc = np.mean(auc_rocs)

        print(f"Mean Accuracy: {mean_acc:.2f}")
        print(f"Mean Precision: {mean_precision:.2f}")
        print(f"Mean Recall: {mean_recall:.2f}")
        print(f"Mean AUC ROC: {mean_auc_roc:.2f}")
        return mean_auc_roc, mean_acc, mean_precision, mean_recall



class PGExplainer_Graph(Explainer):
    def __init__(self, model, num_nodes, args):
        super(PGExplainer_Graph, self).__init__()
        
        hiddens = [int(s) for s in args.hiddens.split('-')]
        hiddensize = hiddens[-1]
        
        self.elayers = nn.Sequential(
            nn.Linear(hiddensize * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        self.model = model
        self.args = args

        self.num_nodes = num_nodes
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)

        rc = torch.arange(num_nodes).unsqueeze(0).repeat(num_nodes, 1)
        self.row = rc.transpose(0, 1).reshape(-1)
        self.col = rc.reshape(-1)
        self.mask_act = 'sigmoid'
        self.shift = args.shift
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-1, b=1) # pos
                # nn.init.uniform_(m.weight, a=-0.001, b=0.001) # neg
                nn.init.constant_(m.bias, 0)

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        if training:
            bias = 0.0  # Adjust bias as needed
            random_noise = torch.rand(log_alpha.shape, device=log_alpha.device) * (1.0 - 2 * bias) + bias
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            # gate_inputs = torch.sigmoid(gate_inputs)
            gate_inputs = shifted_sigmoid(gate_inputs, shift=self.shift)
        else:
            # gate_inputs = torch.sigmoid(log_alpha)
            gate_inputs = shifted_sigmoid(log_alpha, shift=self.shift)
        return gate_inputs

    def forward(self, inputs, training=None):
        x, embed, adj, tmp, label = inputs

        self.label = torch.argmax(label.float(), dim=-1)
        self.tmp = tmp
        
        f1 = embed[self.row]
        f2 = embed[self.col]
        
        f12self = torch.cat([f1, f2], dim=-1)
        
        h = f12self.requires_grad_(True)
        h = self.elayers(h)
        self.h = h

        self.values = h.view(-1)
        values = self.concrete_sample(self.values, beta=tmp, training=training)
        indices = torch.stack([self.row, self.col], dim=0).to(values.device)
        sparsemask = torch.sparse_coo_tensor(indices, values, torch.Size([self.num_nodes, self.num_nodes])).to_dense()
        sym_mask = sparsemask.to_dense()
        self.mask = sym_mask
        
        sym_mask = (sym_mask + sym_mask.transpose(0, 1)) / 2
        if self.args.task=='pos':
            masked_adj = adj * sym_mask.to(adj.device)
        elif self.args.task=='neg':
            masked_adj = adj + (1-adj) * sym_mask.to(adj.device)
        self.masked_adj = masked_adj
        x = x.unsqueeze(0)
        adj = self.masked_adj.unsqueeze(0)

        # with torch.no_grad():
        output = self.model((x, adj))
        res = F.softmax(output, dim=-1)
        return res

    def loss(self, pred, pred_label):
        pred_reduce = pred[0]
        gt_label_node = self.label
        logit = pred_reduce[gt_label_node]
        pred_loss = -torch.log(logit)
        mask = self.mask
        if self.mask_act == "sigmoid":
            # mask = torch.sigmoid(self.mask)
            mask = shifted_sigmoid(self.mask, shift=self.shift)
        elif self.mask_act == "ReLU":
            mask = F.relu(self.mask)

        size_loss = self.args.coff_size * torch.sum(mask)
        mask = mask * 0.99 + 0.005
        mask_ent = -mask * torch.log(mask) - (1.0 - mask) * torch.log(1.0 - mask)
        mask_ent_loss = self.args.coff_ent * torch.mean(mask_ent)

        if self.args.task=='neg':
            pred_loss *= -1
        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def _pos_singlegraph(self, data:SyntheticDataset, nodeid):
        # no remap for ba2motif
        score_edges =[]

        feature = torch.tensor(data.feas[nodeid], device=self.args.device)
        adj = torch.tensor(data.adjs[nodeid])
        emb = data.embs[nodeid]
        label = torch.tensor(data.labels[nodeid])
        tmp = 1.0 # default to 1.0
        
        self((feature, emb, adj, tmp, label))
        mask = self.masked_adj.detach().cpu().numpy()
        
        # adj = coo_matrix(adj)
        num_nodes = self.num_nodes
        for src in range(num_nodes):
            for des in range(num_nodes):
                if not adj[src][des]: # pos
                    continue

                score = mask[src][des].item()
                score_edges.append((score, (src,des)))
        
        score_edges.sort(reverse=True, key=lambda x: x[0])
        return score_edges
    
    def _neg_singlegraph(self, data:SyntheticDataset, nodeid):
        # no remap for ba2motif
        score_edges =[]

        feature = torch.tensor(data.feas[nodeid], device=self.args.device)
        adj = torch.tensor(data.adjs[nodeid])
        emb = data.embs[nodeid]
        label = torch.tensor(data.labels[nodeid])
        tmp = 1.0 # default to 1.0
        
        self((feature, emb, adj, tmp, label))
        mask = self.masked_adj.detach().cpu().numpy()
        # print(mask)
        # adj = coo_matrix(adj)
        num_nodes = self.num_nodes
        for src in range(num_nodes):
            for des in range(num_nodes):
                if adj[src][des]: # neg
                    continue

                score = mask[src][des].item()
                score_edges.append((score, (src,des)))
        
        score_edges.sort(reverse=True, key=lambda x: x[0])
        return score_edges
    
    def get_explanation_scores(self, data:SyntheticDataset):
        """"""
        allnodes = data.nodes
        auc_rocs = []
        for nodeid in tqdm(allnodes):
            if self.args.task == 'neg':
                score_edges = self._neg_singlegraph(data, nodeid)
                edge_reals = self._get_neg_reals(nodeid, score_edges, data)
            elif self.args.task == 'pos':
                score_edges = self._pos_singlegraph(data, nodeid)
                edge_reals = self._get_pos_reals(nodeid, score_edges, data)

            scores = []
            reals = []
            
            for score, edge in score_edges:
                if edge_reals[edge]:
                    reals.append(1)
                else:
                    reals.append(0)
                scores.append(score)

            try:
                rocauc = roc_auc_score(reals, scores)
                auc_rocs.append(rocauc)
            except Exception as e:
                continue
        
        return np.mean(auc_rocs)

class PGExplainer_Node(Explainer):
    def __init__(self, model, elayer_inputdim, args):
        super(PGExplainer_Node, self).__init__()
        self.elayers = nn.Sequential(
            nn.Linear(elayer_inputdim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.model = model
        self.args = args

    def _masked_adj(self, mask, adj):
        sym_mask = (mask + mask.t()) / 2
        adj = torch.tensor(adj.todense(), dtype=torch.float32, device=self.args.device)
        # print(adj.todense())
        # print(sym_mask)
        masked_adj = adj * sym_mask
        diag_mask = torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0])
        return masked_adj * diag_mask.to(masked_adj.device)

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        if training:
            bias = self.args.sample_bias
            random_noise = torch.rand_like(log_alpha, device=self.args.device) * (1 - 2 * bias) + bias
            gate_inputs = torch.log(random_noise) - torch.log(1 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)
        
        return gate_inputs

    def forward(self, inputs, training=False):
        x, adj, nodeid, embed, tmp = inputs
        if isinstance(embed, np.ndarray):
            embed = torch.tensor(embed, dtype=torch.float32, device=self.args.device)
        
        rows = adj.row
        cols = adj.col

        f1 = embed[rows]
        f2 = embed[cols]
        selfemb = embed[nodeid].unsqueeze(0).repeat(f1.shape[0], 1)

        f12self = torch.cat([f1, f2, selfemb], dim=-1)

        h = f12self
        h.requires_grad_(True)
        h = self.elayers(h)
        self.values = h.view(-1)
        values = self.concrete_sample(self.values, beta=tmp, training=training)
        
        mask = self.sparse_to_dense(rows, cols, values, adj.shape)
        masked_adj = self._masked_adj(mask, adj)

        self.mask = mask
        self.masked_adj = masked_adj

        output = self.model((x, masked_adj))

        node_pred = output[nodeid, :]
        res = F.softmax(node_pred, dim=0)
        return res

    def sparse_to_dense(self, rows, cols, values, shape):
        index_tensor = np.stack([rows, cols],axis=0)
        # index_tensor = torch.stack([rows, cols], dim=0)
        sparse_tensor = torch.sparse.FloatTensor(torch.tensor(index_tensor, dtype=torch.int64, device=self.args.device), values, torch.Size(shape))
        return sparse_tensor.to_dense()

    def loss(self, pred, pred_label, label, node_idx, adj_tensor=None):
        label = torch.argmax(label, dim=-1)
        pred_label_node = pred_label[node_idx]
        logit = pred[pred_label_node]
        logit = logit + 1e-6
        pred_loss = -torch.log(logit)

        if self.args.budget <=0:
            size_loss = self.args.coff_size * torch.sum(self.mask)
        else:
            size_loss = self.args.coff_size * F.relu(torch.sum(self.mask) - self.args.budget)

        scale=0.99
        mask = self.mask*(2*scale-1.0)+(1.0-scale)

        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.args.coff_ent * mask_ent.mean()

        l2norm = sum(torch.norm(p) for p in self.elayers.parameters())
        l2norm *= self.args.weight_decay
        total_loss = pred_loss + size_loss + l2norm + mask_ent_loss
        
        if self.args.budget > 0 and self.args.coff_connect > 0:
            adj_tensor_dense = adj_tensor.to_dense()
            noise = torch.rand(adj_tensor_dense.shape) * 0.001
            adj_tensor_dense += noise
            cols = torch.argsort(adj_tensor_dense, descending=True, dim=-1)
            sampled_rows = torch.arange(adj_tensor_dense.shape[0]).unsqueeze(-1)
            sampled_cols_0 = cols[:, 0].unsqueeze(-1)
            sampled_cols_1 = cols[:, 1].unsqueeze(-1)
            sampled0 = torch.cat((sampled_rows, sampled_cols_0), -1)
            sampled1 = torch.cat((sampled_rows, sampled_cols_1), -1)
            sample0_score = mask[sampled0[:, 0], sampled0[:, 1]]
            sample1_score = mask[sampled1[:, 0], sampled1[:, 1]]
            connect_loss = torch.sum(-(1.0 - sample0_score) * torch.log(1.0 - sample1_score) - sample0_score * torch.log(sample1_score))
            connect_loss *= self.args.coff_connect
            total_loss += connect_loss
        
        # return total_loss, pred_loss, size_loss
        return total_loss

    def _pos_singlegraph(self, data:SyntheticDataset, nodeid):
        score_edges=[]
        nodeid = data.remap[nodeid]

        feature = data.sub_features[nodeid]
        adj = data.sub_adjs[nodeid]
        emb = data.sub_embeds[nodeid]
        label = data.sub_label_tensors[nodeid]
        tmp = 1.0 # default to 1.0
        # x, adj, nodeid, embed, tmp 
        self.__call__((feature, adj, 0,emb, tmp))
        mask = self.masked_adj.detach().cpu().numpy()
        num_nodes = adj.shape[-1]
        adj=adj.todense()
        for src in range(num_nodes):
            for des in range(num_nodes):
                if not adj[src,des]: # pos
                    continue

                score = mask[src][des].item()
                score_edges.append((score, (src,des)))
        
        score_edges.sort(reverse=True, key=lambda x: x[0])
        return score_edges
    
    def _neg_singlegraph(self, data:SyntheticDataset, nodeid):
        score_edges=[]
        nodeid = data.remap[nodeid]

        feature = data.sub_features[nodeid]
        adj = data.sub_adjs[nodeid]
        emb = data.sub_embeds[nodeid]
        label = data.sub_label_tensors[nodeid]
        tmp = 1.0 # default to 1.0
        
        self.__call__((feature, adj, 0,emb, tmp))
        mask = self.masked_adj.detach().cpu().numpy()
        num_nodes = adj.shape[-1]
        adj=adj.todense()
        for src in range(num_nodes):
            for des in range(num_nodes):
                if adj[src, des]: # neg
                    continue

                score = mask[src][des].item()
                score_edges.append((score, (src,des)))
        
        score_edges.sort(reverse=True, key=lambda x: x[0])
        return score_edges
    
    def get_explanation_scores(self, data:SyntheticDataset):
        """"""
        allnodes = data.nodes
        auc_rocs = []
        for nodeid in tqdm(allnodes):
            if self.args.task == 'neg':
                score_edges = self._neg_singlegraph(data, nodeid)
                edge_reals = self._get_neg_reals(nodeid, score_edges, data)
            elif self.args.task == 'pos':
                score_edges = self._pos_singlegraph(data, nodeid)
                edge_reals = self._get_pos_reals(nodeid, score_edges, data)

            scores = []
            reals = []
            
            for score, edge in score_edges:
                if edge_reals[edge]:
                    reals.append(1)
                else:
                    reals.append(0)
                scores.append(score)

            try:
                rocauc = roc_auc_score(reals, scores)
                auc_rocs.append(rocauc)
            except Exception as e:
                continue
        
        return np.mean(auc_rocs)



class GraphCFE(Explainer):
    def __init__(self, args, model):
        """_summary_

        Args:
            init_params (_type_):
                vae_type, init_params, u_dim(1?), max_num_nodes
            args (_type_):
                Required
                -h_dim, z_dim, u_dim, encoder_type, graph_pool_type, disable_u

                Already existing
                -dropout, device
        """
        super(GraphCFE, self).__init__()
        self.args = args
        self.model = model
        # self.vae_type = init_params['vae_type']  # graphVAE
        # self.x_dim = init_params['x_dim']
        # self.vae_type = args.vae_type  # graphVAE? 없어도될듯
        self.x_dim = args.x_dim         # 10?feature dim같은데
        self.h_dim = args.dim_h         # 16쯤
        self.z_dim = args.dim_z         # 16
        self.u_dim = 1 # init_params['u_dim']
        self.dropout = args.dropout
        # self.max_num_nodes = init_params['max_num_nodes']
        self.max_num_nodes = args.max_num_nodes
        self.encoder_type = 'gcn'
        self.graph_pool_type = 'mean'
        self.disable_u = args.disable_u
        self.device = args.device
        self.bn = args.explainer_bn

        if self.disable_u:
            self.u_dim = 0
            print('disable u!')
        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)

        # prior
        self.prior_mean = MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=self.device)
        self.prior_var = nn.Sequential(MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=self.device), nn.Sigmoid())

        if self.args.gnn_task == 'graph':
            self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
            self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU(), nn.Sigmoid())
            self.decoder_x = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                           nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                           nn.Linear(self.h_dim, self.max_num_nodes*self.x_dim))
            self.decoder_a = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                           nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                           nn.Linear(self.h_dim, self.max_num_nodes*self.max_num_nodes), nn.Sigmoid())
            self.graph_norm = nn.BatchNorm1d(self.h_dim)
        else:
            self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.ReLU())
            self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.ReLU(), nn.Sigmoid())
            self.decoder_x = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                           nn.Linear(self.h_dim, self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                           nn.Linear(self.h_dim, self.x_dim))
            self.decoder_a = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                           nn.Linear(self.h_dim, self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                           nn.Linear(self.h_dim, self.max_num_nodes), nn.Sigmoid())
            self.graph_norm = nn.Identity()

    def encoder(self, features, u, adj, y_cf):
        # Q(Z|X,U,A,Y^CF)
        # input: x, u, A, y^cf
        # output: z
        task = self.args.gnn_task

        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        if task == 'graph':
            graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)  # n x h_dim
        else:
            """"""
            # graph_rep = graph_rep.squeeze(0)
        
        
        # graph_rep = self.graph_norm(graph_rep)
        if self.disable_u:
            if task == 'graph':
                # 이거 그래프도 dim=-1로 해도되지않나
                z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=1))
                z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=1))
            else:
                z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=-1))
                z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=-1))
        else:
            z_mu = self.encoder_mean(torch.cat((graph_rep, u, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, u, y_cf), dim=1))

        return z_mu, z_logvar

    def get_represent(self, features, u, adj, y_cf):
        u_onehot = u
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)

        return z_mu, z_logvar

    def decoder(self, z, y_cf, u):
        task = self.args.gnn_task

        if task=='graph':
            if self.disable_u:
                adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes,
                                                                                    self.max_num_nodes)
            else:
                adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes,
                                                                                    self.max_num_nodes)
            features_reconst = self.decoder_x(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes, self.x_dim)
        elif task=='node':
            # z     [B(1), max_num_nodes, z_hidden]
            # y_cf  [B(1), max_num_nodes, 1]
            # 1, 24, 17 
            if self.disable_u:
                adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=-1)).view(-1, self.max_num_nodes,self.max_num_nodes)
            else:
                adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=-1)).view(-1, self.max_num_nodes,
                                                                                    self.max_num_nodes)
            features_reconst = self.decoder_x(torch.cat((z, y_cf), dim=-1)).view(-1, self.max_num_nodes, self.x_dim)

        return features_reconst, adj_reconst

    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def prior_params(self, u):  # P(Z|U)
        if self.disable_u:
            z_u_mu = torch.zeros((len(u),self.h_dim)).to(self.device)
            z_u_logvar = torch.ones((len(u),self.h_dim)).to(self.device)
        else:
            z_u_logvar = self.prior_var(u)
            z_u_mu = self.prior_mean(u)
        return z_u_mu, z_u_logvar

    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def score(self):
        return

    def forward(self, features, u, adj, y_cf):
        u_onehot = u
        z_u_mu, z_u_logvar = self.prior_params(u_onehot)
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)
        # reparameterize
        z_sample = self.reparameterize(z_mu, z_logvar)
        # decoder
        features_reconst, adj_reconst = self.decoder(z_sample, y_cf, u_onehot)

        return {'z_mu': z_mu, 'z_logvar': z_logvar, 'adj_permuted': adj, 'features_permuted': features,
                'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'z_u_mu': z_u_mu, 'z_u_logvar': z_u_logvar}

    def compute_loss(self, params):
        """"""
        pred_model, z_mu, z_logvar, adj_permuted, features_permuted, adj_reconst, features_reconst, \
        adj_input, features_input, y_cf, z_u_mu, z_u_logvar, z_mu_cf, z_logvar_cf = params['pred_model'], params['z_mu'], \
            params['z_logvar'], params['adj_permuted'], params['features_permuted'], params['adj_reconst'], params['features_reconst'], \
            params['adj_input'], params['features_input'], params['y_cf'], params['z_u_mu'], params['z_u_logvar'], params['z_mu_cf'], params['z_logvar_cf']

        # kl loss
        loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
        loss_kl = torch.mean(loss_kl)

        # similarity loss
        size = len(features_permuted) # batchsize
        
        dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
        dist_a = distance_graph_prob(adj_permuted, adj_reconst)

        beta = 10

        loss_sim = beta * dist_x + 10 * dist_a

        # CFE loss
        pred_model_name = pred_model.__class__.__name__
        if pred_model_name == 'GCN_Node':
            y_pred = pred_model((features_reconst.squeeze(0), adj_reconst.squeeze(0)))  # num_nodes x num_class
            y_pred = y_pred.unsqueeze(0).permute(0,2,1) # squeeze했으니까 다시 unsqueeze
            y_cf = y_cf.squeeze(-1)
            # sft = F.softmax(y_pred, dim=1)
            loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=1), y_cf.long()) # cf랑비슷->반대로

        elif pred_model_name == 'GCN_Graph':
            y_pred = pred_model((features_reconst, adj_reconst))  # B x num_class
            loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.view(-1).long())
        else:
            raise Exception(f"Unknown pred_model '{pred_model}'. Please refer to models/models.py for valid options.")
        
        # rep loss
        if z_mu_cf is None:
            loss_kl_cf = 0.0
        else:
            loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
            loss_kl_cf = torch.mean(loss_kl_cf)

        loss = 1. * loss_sim + 1 * loss_kl + 1.0 * loss_cfe
        loss_results = {'loss': loss, 'loss_kl': loss_kl, 'loss_sim': loss_sim, 'loss_cfe': loss_cfe, 'loss_kl_cf':loss_kl_cf}
        return loss_results

    def _pos_singlegraph(self, data:SyntheticDataset, nodeid):
        score_edges = []
        self.eval()
        device = self.device
        args = self.args
        task = args.task

        target_nodeid = 0 # args or param으로
        if args.gnn_task == 'node':
            remap = data.remap
            remapped_nodeid = remap[nodeid]

            features = data.sub_features[remapped_nodeid].unsqueeze(0)
            support = data.sub_support_tensors[remapped_nodeid].unsqueeze(0)
            edge_labels = np.array(data.sub_edge_labels[remapped_nodeid].todense())
            labels = torch.argmax(data.sub_label_tensors[remapped_nodeid].cpu().detach(), axis=1).to(dtype=torch.long, device=device).unsqueeze(0)
            num_nodes = support.size()[-1]

            o_features, o_support = features.squeeze(0), support.to_dense().squeeze(0).to_sparse()
            features, support, labels = data.pad_graph((features, support, labels), args.max_num_nodes)
            u = torch.zeros((1, 1), dtype=torch.float32, device=device)
            y_cf = (1 - labels).unsqueeze(-1)
            
            model_return = self.__call__(features, u, support.to_dense(), y_cf)
            adj_recon = model_return['adj_reconst'].squeeze(0)
            adj = support.squeeze(0).to_dense()
            label = labels.squeeze(0)[target_nodeid]

            # 이거 여기서하는게 맞나 함수 호출하기전에하는게 맞나
            if label.item() == 0: # deletion인데 motif 없는거
                return []

            # original_output = model((o_features, o_support)) # treecycle일때만
            for src in range(num_nodes):
                for des in range(num_nodes):
                    if not adj[src][des]: # pos
                        continue

                    score = adj_recon[src][des].item()
                    score_edges.append((score, (src,des)))

        elif args.gnn_task == 'graph':
            """"""

        score_edges.sort(reverse=True, key=lambda x: x[0])
        # print(score_edges)
        return score_edges
    
    def _pos_batchedgraph(self, data:SyntheticDataset, batch):
        """
        Just for efficiency when forwarding
        nodeid ~ nodeid+batchsize-1

        !! Applicable only for graph classification
        !! node classification도 어차피 padding할거면 batch로 될거같은데

        Args:
            batch(List): [nodeid, ...]

        Returns:
            score_edges_batch(List): [score_edges]
                score_edges(List): (score, edge)
                score(float): 
                edge(tuple): (src, des)
        """
        score_edges_batch = []
        self.eval()
        device = self.device
        args = self.args
        task = args.task

        target_nodeid = 0
        if args.gnn_task == 'node':
            """"""
            raise NotImplementedError
        elif args.gnn_task == 'graph':
            num_nodes = data.adjs.shape[-1]
            features_batch = torch.tensor([data.feas[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
            adj_batch = torch.tensor([data.adjs[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
            u_batch = torch.zeros((len(batch), 1), dtype=torch.float32, device=device)
            labels_batch = torch.tensor([np.argmax(data.labels[nodeid]) for nodeid in batch], dtype=torch.long, device=device)

            y_cf_batch = (1 - labels_batch).unsqueeze(1)
            model_return = self.__call__(features_batch, u_batch, adj_batch, y_cf_batch)

            adj_recon_batch = model_return['adj_reconst']

            
            for b_idx, nodeid in enumerate(batch):
                score_edges = []
                adj_recon = adj_recon_batch[b_idx]
                adj = adj_batch[b_idx]
                label = labels_batch[b_idx]
                
                for src in range(num_nodes):
                    for des in range(num_nodes):
                        if not adj[src][des]: # pos
                            continue

                        score = adj_recon[src][des].item()
                        score_edges.append((score, (src,des)))
                
                
                score_edges.sort(reverse=True, key=lambda x: x[0])
                score_edges_batch.append(score_edges)

        return score_edges_batch
    
    def _neg_batchedgraph(self, data:SyntheticDataset, batch):
        """"""
        score_edges_batch = []
        self.eval()
        device = self.device
        args = self.args
        task = args.task

        target_nodeid = 0
        if args.gnn_task == 'node':
            """"""
            raise NotImplementedError
        elif args.gnn_task == 'graph':
            num_nodes = data.adjs.shape[-1]
            features_batch = torch.tensor([data.feas[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
            adj_batch = torch.tensor([data.adjs[nodeid] for nodeid in batch], dtype=torch.float32, device=device)
            u_batch = torch.zeros((len(batch), 1), dtype=torch.float32, device=device)
            labels_batch = torch.tensor([np.argmax(data.labels[nodeid]) for nodeid in batch], dtype=torch.long, device=device)

            y_cf_batch = (1 - labels_batch).unsqueeze(1)
            model_return = self.__call__(features_batch, u_batch, adj_batch, y_cf_batch)

            adj_recon_batch = model_return['adj_reconst']

            for b_idx, nodeid in enumerate(batch):
                score_edges = []
                adj_recon = adj_recon_batch[b_idx]
                adj = adj_batch[b_idx]
                label = labels_batch[b_idx]
                
                for src in range(num_nodes):
                    for des in range(num_nodes):
                        if adj[src][des]: # neg, pos랑 여기만 다름
                            continue

                        score = adj_recon[src][des].item()
                        score_edges.append((score, (src,des)))
                
                
                score_edges.sort(reverse=True, key=lambda x: x[0])
                score_edges_batch.append(score_edges)

        return score_edges_batch

    def _neg_singlegraph(self, data:SyntheticDataset, nodeid):
        score_edges = []
        self.eval()
        device = self.device
        args = self.args
        task = args.task
        dataset_name = args.dataset

        target_nodeid = 0 # args or param으로
        if dataset_name == 'syn3':
            remap = data.remap
            remapped_nodeid = remap[nodeid]

            features = data.sub_features[remapped_nodeid].unsqueeze(0)
            support = data.sub_support_tensors[remapped_nodeid].unsqueeze(0)
            labels = torch.argmax(data.sub_label_tensors[remapped_nodeid].cpu().detach(), axis=1).to(dtype=torch.long, device=device).unsqueeze(0)
            num_nodes = support.size()[-1]

            features, support, labels = data.pad_graph((features, support, labels), args.max_num_nodes)
            u = torch.zeros((1, 1), dtype=torch.float32, device=device)
            y_cf = (1 - labels).unsqueeze(-1)
            
            model_return = self.__call__(features, u, support.to_dense(), y_cf)
            adj_recon = model_return['adj_reconst'].squeeze(0)

            adj = support.squeeze(0).to_dense()
            label = labels.squeeze(0)[target_nodeid]

            if label.item() == 1: # addition인데 motif 있는거
                return []

            for src in range(num_nodes):
                for des in range(num_nodes):
                    if adj[src][des]: # neg
                        continue

                    score = adj_recon[src][des].item()
                    score_edges.append((score, (src,des)))

        elif dataset_name == 'syn4':
            remap = data.remap
            remapped_nodeid = remap[nodeid]

            features = data.sub_features[remapped_nodeid]
            support = data.sub_support_tensors[remapped_nodeid]
            labels = torch.argmax(data.sub_label_tensors[remapped_nodeid].cpu().detach(), axis=1).to(dtype=torch.long, device=device)
            num_nodes = support.size()[-1]
            
            label = labels[target_nodeid]
            if label.item() == 0: # syn4 는 무조건 motif 있어야함
                return []

            indices = support.coalesce().indices()
            values = support.coalesce().values()
            size = support.coalesce().size()

            # 랜덤하게 지울거 고르고
            deletion_candidates = []
            for edge_index in range(len(values)):
                src, des = indices[0][edge_index].item(), indices[1][edge_index].item()
                if src >= des:
                    continue
                if data.sub_edge_labels[data.remap[nodeid]].todense()[src,des] or data.sub_edge_labels[data.remap[nodeid]].todense()[des,src]:
                    deletion_candidates.append((src, des))
            edges_to_delete = random.sample(deletion_candidates, 1) # 일단 k=1 고정
            
            for src_del, des_del in edges_to_delete:
                mask = ~(((indices[0] == src_del) & (indices[1] == des_del)) | ((indices[0] == des_del) & (indices[1] == src_del)))
                new_indices = indices[:, mask]
                new_values = values[mask]

                # motif에서 edge 한개제거
                features = features.unsqueeze(0)
                support = torch.sparse_coo_tensor(new_indices, new_values, size).unsqueeze(0)
                labels = torch.zeros_like(labels).unsqueeze(0)

                features, support, labels = data.pad_graph((features, support, labels), args.max_num_nodes)
                u = torch.zeros((1, 1), dtype=torch.float32, device=device)
                y_cf = (1 - labels).unsqueeze(-1)
                
                model_return = self.__call__(features, u, support.to_dense(), y_cf)
                adj_recon = model_return['adj_reconst'].squeeze(0)
                adj = support.squeeze(0).to_dense()

                for src in range(num_nodes):
                    for des in range(num_nodes):
                        if adj[src][des]: # neg
                            continue

                        score = adj_recon[src][des].item()
                        score_edges.append((score, (src,des)))


        elif dataset_name == 'BA-2motif':
            """"""
            raise NotImplementedError
      
        score_edges.sort(reverse=True, key=lambda x: x[0])
        return score_edges
    
    def get_explanation_scores(self, data:SyntheticDataset=None):
        """
        Not for training the explainer itself, but for testing it.

        Returns:
            _type_: _description_
        """
        if data is None:
            raise ValueError("The 'data' parameter must be provided.")
        scores = []
        reals = []
        auc_rocs = []

        if self.args.gnn_task == 'graph': # 일단 graph classification은 batch로 처리
            nodes = data.nodes
            batch_size = 1024
            all_batches = [nodes[i:i + batch_size] for i in range(0, len(nodes), batch_size)]
            
            for batch in all_batches:
                if self.args.task == 'neg':
                    score_edges_batch = self._neg_batchedgraph(data, batch)
                elif self.args.task == 'pos':
                    score_edges_batch = self._pos_batchedgraph(data, batch)
                
                for batch_idx, nodeid in enumerate(batch):
                    score_edges = score_edges_batch[batch_idx]
                    if self.args.task == 'neg':
                        edge_reals = self._get_neg_reals(nodeid, score_edges, data)
                    elif self.args.task == 'pos':
                        edge_reals = self._get_pos_reals(nodeid, score_edges, data)
                    scores = []
                    reals = []
                    for score, edge in score_edges:
                        if edge_reals[edge]:
                            reals.append(1)
                        else:
                            reals.append(0)
                        scores.append(score)

                    try:
                        rocauc = roc_auc_score(reals, scores)
                        auc_rocs.append(rocauc)
                    except Exception as e:
                        continue
            return np.mean(auc_rocs)
        elif self.args.gnn_task == 'node':
            allnodes = data.nodes
            for nodeid in tqdm(allnodes):
                if self.args.task == 'neg':
                    score_edges = self._neg_singlegraph(data, nodeid)
                    edge_reals = self._get_neg_reals(nodeid, score_edges, data)
                elif self.args.task == 'pos':
                    score_edges = self._pos_singlegraph(data, nodeid)
                    edge_reals = self._get_pos_reals(nodeid, score_edges, data)

                scores = []
                reals = []
                for score, edge in score_edges:
                    if edge_reals[edge]:
                        reals.append(1)
                    else:
                        reals.append(0)
                    scores.append(score)

                try:
                    rocauc = roc_auc_score(reals, scores)
                    auc_rocs.append(rocauc)
                    # print(f'num edges : {data.sub_edge_labels[data.remap[nodeid]].nnz}, auc roc : {rocauc}')
                    # print(f'\nauc roc : {rocauc}')
                except Exception as e:
                    # print(e)
                    continue
                # print(np.mean(auc_rocs))
            
            return np.mean(auc_rocs)



class GCN_NodePerturb(nn.Module):
    """
    GCN used in CF-GNN Explainer
    """
    def __init__(self, args, graph, num_classes, edge_additions=False):
        super(GCN_NodePerturb, self).__init__()
        """
        adj 하나 받는거 대신에 그냥 data랑 nodeid를 받는게 낫나
        edge_additions = True for neg, False for pos?
        """
        features, adj = graph
        self.args = args
        self.edge_additions = edge_additions
        self.adj = adj
        self.num_nodes = self.adj.shape[0]
        self.num_classes = num_classes
        self.nfeat = features.shape[-1]
        # hiddens = [int(s) for s in self.args.hiddens.split('-')]
        self.beta = args.beta
        self.dropout = args.dropout
        self.target_nodeid = 0
        
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2)  + self.num_nodes

        # 초기값..
        #### P == mask ####
        if self.edge_additions: # neg
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else: # pos
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        self.reset_parameters()

        # self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
        # self.gc2 = GraphConvolutionPerturb(nhid, nhid)
        # self.gc3 = GraphConvolution(nhid, nout)
        # self.pred_layer = nn.Linear(nhid + nhid + nout, nclass)
        # self.dropout = dropout

        #### original pretrained model ####
        # ㄴ이거 require grad false로 해도 P_vec이 학습이 되나
        # relu, bias=True
        # nfeat-20, 20-20, 20-20, 60-num_classes(concat)
        # bn도 있어야되나
        # 이렇게 하지말고 그냥 GCN_Node 하나 만드는게 나을듯
        # self.gc1 = GraphConvolution_Node(args, self.nfeat, hiddens[0], activation=F.relu, bias=True)
        # self.gc2 = GraphConvolution_Node(args, hiddens[1], hiddens[1], activation=F.relu, bias=True)
        # self.gc3 = GraphConvolution_Node(args, hiddens[2], hiddens[2], activation=F.relu, bias=True)
        # self.pred_layer = nn.Linear(hiddens[0] + hiddens[1] + hiddens[2], self.num_classes)

        self.model = GCN_Node(args=args, input_dim=self.nfeat, output_dim=self.num_classes).to(args.device)
        self.model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))

    @torch.no_grad()
    def reset_parameters(self, eps=10**-4):
        if self.edge_additions:
            adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).to(self.P_vec.device)
            for i in range(len(adj_vec)):
                if i < 1:
                    adj_vec[i] = adj_vec[i] - eps
                else:
                    adj_vec[i] = adj_vec[i] + eps
            torch.add(self.P_vec, adj_vec)       #self.P_vec is all 0s
        else:
            torch.sub(self.P_vec, eps)

    def forward(self, x, sub_adj):
        device = x.device
        self.sub_adj = sub_adj
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry

        A_tilde = torch.tensor(0.0, device=device).repeat(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        if self.edge_additions:         # Learn new adj matrix directly
            # A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes, device=device)
            A_tilde = F.sigmoid(self.P_hat_symm)
        else:       # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            # A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes, device=device)
            A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj

        D_tilde = get_degree_matrix(A_tilde).detach()       # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.matmul(torch.matmul(D_tilde_exp, A_tilde), D_tilde_exp).to(x.device)
        
        # A_tilde 처음 0.7쯤? sigmoid(1)=0.7
        out = self.model((x, A_tilde)) # 원래도 norm_adj 안했
        # out = self.model((x, norm_adj))
        return F.log_softmax(out, dim=1)

    def forward_prediction(self, x):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        # inference할때 쓰는거 training할때는 안씀
        device = x.device
        # print(self.P_hat_symm)
        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()      # threshold P_hat

        if self.edge_additions:
            # A_tilde = self.P + torch.eye(self.num_nodes, device=device)
            A_tilde = self.P
        else:
            # A_tilde = self.P * self.adj + torch.eye(self.num_nodes, device=device)
            A_tilde = self.P * self.adj

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.matmul(torch.matmul(D_tilde_exp, A_tilde), D_tilde_exp)

        # x = self.model((x, norm_adj))
        x = self.model((x, A_tilde)) # 원래도 norm_adj 안했
        return F.log_softmax(x, dim=1), self.P

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        y_pred_orig = y_pred_orig[self.target_nodeid]
        pred_same = (y_pred_new_actual == y_pred_orig).float()
        # pred_same = 1.0

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)
        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        
        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = -F.nll_loss(output, y_pred_orig)
        # print(F.softmax(output[0], dim=-1))
        # loss_pred = torch.log(F.softmax(output[0], dim=-1)[1]) # 멀어지도록
        # loss_pred = torch.log(F.softmax(output[0], dim=-1)[y_pred_orig[0].item()]) # 멀어지도록
        # print(loss_pred)
        # print(output)
        # print(y_pred_orig)
        # print('-----')
        loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) / 2      # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj

class GCN_GraphPerturb(nn.Module):
    """
    GCN used in CF-GNN Explainer
    """
    def __init__(self, args, graph, num_classes, edge_additions=False):
        super(GCN_GraphPerturb, self).__init__()
        features, adj = graph
        self.args = args
        self.edge_additions = edge_additions
        self.adj = adj
        self.num_nodes = self.adj.shape[0]
        self.num_classes = num_classes
        self.nfeat = features.shape[-1]
        # hiddens = [int(s) for s in self.args.hiddens.split('-')]
        self.beta = args.beta
        self.dropout = args.dropout
        
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2)  + self.num_nodes

        # 초기값..
        #### P == mask ####
        if self.edge_additions:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        self.reset_parameters()

        self.model = GCN_Graph(args=args, input_dim=self.nfeat, output_dim=self.num_classes).to(args.device)
        self.model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))
    
    @torch.no_grad()
    def reset_parameters(self, eps=10**-4):
        if self.edge_additions:
            adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).to(self.P_vec.device)
            for i in range(len(adj_vec)):
                if i < 1:
                    adj_vec[i] = adj_vec[i] - eps
                else:
                    adj_vec[i] = adj_vec[i] + eps
            torch.add(self.P_vec, adj_vec)       #self.P_vec is all 0s
        else:
            torch.sub(self.P_vec, eps)

    def forward(self, x, sub_adj):
        device = x.device
        self.sub_adj = sub_adj
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry

        A_tilde = torch.tensor(0.0, device=device).repeat(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True
        if self.edge_additions:         # Learn new adj matrix directly
            # A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes, device=device)
            A_tilde = F.sigmoid(self.P_hat_symm)
        else:       # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            # A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes, device=device) 
            A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj

        # D_tilde = get_degree_matrix(A_tilde).detach()       # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        # D_tilde_exp = D_tilde ** (-1 / 2)
        # D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
        
        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        # norm_adj = torch.matmul(torch.matmul(D_tilde_exp, A_tilde), D_tilde_exp).to(x.device)

        out = self.model((x.unsqueeze(0), A_tilde.unsqueeze(0))) # 원래도 norm_adj 안했
        # out = self.model((x.unsqueeze(0), norm_adj.unsqueeze(0)))
        return F.log_softmax(out, dim=1)

    def forward_prediction(self, x):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        device = x.device
        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()      # threshold P_hat
        if self.edge_additions:
            # A_tilde = self.P + torch.eye(self.num_nodes, device=device)
            A_tilde = self.P
        else:
            # A_tilde = self.P * self.adj + torch.eye(self.num_nodes, device=device)
            A_tilde = self.P * self.adj

        # D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        # D_tilde_exp = D_tilde ** (-1 / 2)
        # D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        # norm_adj = torch.matmul(torch.matmul(D_tilde_exp, A_tilde), D_tilde_exp)

        # print()
        # out = self.model((x.unsqueeze(0), norm_adj.unsqueeze(0)))
        out = self.model((x.unsqueeze(0), A_tilde.unsqueeze(0))) # 원래도 norm_adj 안했
        return F.log_softmax(out, dim=1), self.P

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()
        # pred_same = 1.0
        
        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        # y_pred_orig = y_pred_orig.unsqueeze(0)
        
        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        
        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        loss_graph_dist = sum(sum(abs(cf_adj- self.adj))) / 2      # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj

class CFGNNExplainer(Explainer):
    def __init__(self, args, model, graph, num_classes):
        super(CFGNNExplainer, self).__init__()
        self.model = model
        self.args = args
        self.device = args.device

        features, adj, labels = graph
        self.adj = adj # 여기서 애초에 syn4일때 한개없애버리면 끝?
        self.features = features
        self.labels = labels
        with torch.no_grad():
            if self.args.gnn_task == 'graph': # 그래프일때만 unsqueeze
                features = features.unsqueeze(0)
                adj = adj.unsqueeze(0)
            y_pred_orig = torch.argmax(self.model((features, adj)), dim=1) # target nodeid 0
        self.y_pred_orig = y_pred_orig
        self.num_classes = num_classes

        # 이거 require grad를 false로 하거나
        # optimizer에 p_vec만 주거나?
        # def __init__(self, args, graph, num_classes, edge_additions=False)
        if args.task == 'pos':
            edge_additions = False
        elif args.task == 'neg':
            edge_additions = True

        if args.gnn_task == 'node':
            self.cf_model = GCN_NodePerturb(args, (self.features, self.adj),
                                                self.num_classes, edge_additions=edge_additions)
        elif args.gnn_task == 'graph':
            self.cf_model = GCN_GraphPerturb(args, (self.features, self.adj),
                                                self.num_classes, edge_additions=edge_additions)
            
        for name, param in self.cf_model.named_parameters():
            if not name.startswith("P"): # 마스크만 학습
                param.requires_grad = False

        # for name, param in self.model.named_parameters():
        #     print("orig model requires_grad: ", name, param.requires_grad)
        # for name, param in self.cf_model.named_parameters():
            # print("cf model requires_grad: ", name, param.requires_grad)

    def _train_one_epoch(self, epoch):
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        
        output = self.cf_model.forward(self.x, self.A_x) # mask and forward
        output_actual, self.P = self.cf_model.forward_prediction(self.x)
        
        # Need to use new_idx from now on since sub_adj is reindexed # 이걸 안해도되지않나
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
        # print(loss_pred, loss_graph_dist)
        loss_total.backward()
        nn.utils.clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()
        
        # return (cf_stats, loss_total.item()) # original
        return cf_adj.detach().cpu().numpy(), loss_total.item(), self.cf_model.P_hat_symm

    def _pos_singlegraph(self, nodeid, target_nodeid):
        score_edges = []

        _, adj_score = self.explain(args.cf_optimizer, nodeid, target_nodeid, args.lr, args.n_momentum, args.epochs)
        num_nodes = self.cf_model.num_nodes
        adj = self.adj
        
        for src in range(num_nodes):
            for des in range(num_nodes):
                if src==des or not adj[src][des]:
                    continue
                score = adj_score[src][des].item()
                score_edges.append((score, (src,des)))
        score_edges.sort(reverse=True, key=lambda x: x[0])
        # print(score_edges)
        return score_edges

    def _neg_singlegraph(self, nodeid, target_nodeid):
        score_edges = []

        _, adj_score = self.explain(args.cf_optimizer, nodeid, target_nodeid, args.lr, args.n_momentum, args.epochs)
        num_nodes = self.cf_model.num_nodes
        adj = self.adj
        
        for src in range(num_nodes):
            for des in range(num_nodes):
                if src==des or adj[src][des]:
                    continue
                score = adj_score[src][des].item()
                score_edges.append((score, (src,des)))
        return score_edges
    
    def explain(self, cf_optimizer:str, nodeid, new_idx, lr, n_momentum, num_epochs):
        """_summary_

        Args:
            cf_optimizer (str): _description_
            nodeid (_type_): original nodeid
            new_idx (_type_): 0 (target nodeid)
            lr (_type_): _description_
            n_momentum (_type_): _description_
            num_epochs (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.nodeid = nodeid
        self.new_idx = new_idx

        self.x = self.features
        self.A_x = self.adj
        self.D_x = get_degree_matrix(self.A_x)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
            # self.cf_optimizer = optim.SGD([self.cf_model.P_vec], lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)


        best_adj_score = None
        adj_recons = []
        best_loss = np.inf
        num_cf_examples = 0
        for epoch in range(num_epochs):
            adj_recon, loss_total, adj_score = self._train_one_epoch(epoch)
            if loss_total <= best_loss:
                adj_recons.append(adj_recon)
                best_loss = loss_total
                num_cf_examples += 1
                best_adj_score = adj_score
        # print("{} CF examples for nodeid = {}".format(num_cf_examples, self.nodeid))
        # print(" ")
        return adj_recons, best_adj_score

    def get_explanation_score(self, nodeid, data, target_nodeid=0):
        task = self.args.task
        if task == 'pos':
            score_edges = self._pos_singlegraph(nodeid, target_nodeid)
            edge_reals = self._get_pos_reals(nodeid, score_edges, data)
        elif task == 'neg':
            score_edges = self._neg_singlegraph(nodeid, target_nodeid)
            edge_reals = self._get_neg_reals(nodeid, score_edges, data)
        
        scores = []
        reals = []
        for score, edge in score_edges:
            if edge_reals[edge]:
                reals.append(1)
            else:
                reals.append(0)
            scores.append(score)
        try:
            rocauc = roc_auc_score(reals, scores)
        except Exception as e:
            # print(e)
            rocauc = -1

        return rocauc
    


class ParamExplainer(nn.Module):
	''' The parametric explainer takes node embeddings and condition vector as inputs, 
	and predicts edge importance scores. Constructed as a 2-layer MLP.
	Args:
		embed_dim: Integer. Dimension of node embeddings.
		graph_level: Boolean. Whether to explain a graph-level prediction task or 
		node-level prediction task.
		hidden_dim: Integer. Hidden dimension of the MLP in the explainer.
	'''

	def __init__(self, embed_dim: int, graph_level: bool, hidden_dim: int = 600):
		super(ParamExplainer, self).__init__()

		self.embed_dims = embed_dim * (2 if graph_level else 3)
		self.cond_dims = embed_dim

		self.emb_linear1 = nn.Sequential(nn.Linear(self.embed_dims, hidden_dim), nn.ReLU())
		self.emb_linear2 = nn.Linear(hidden_dim, 1)

		self.cond_proj = nn.Sequential(nn.Linear(self.cond_dims, self.embed_dims), nn.ReLU())

	def forward(self, embed, cond):
		'''
		Args:
			embeds: Tensor of shape [n_edges, 2*embed_dim] or [n_edges, 3*embed_dim*].
			cond: Tensor of shape [1, embed_dim]. Condition vector.
		'''
		# print(cond.shape, embed.shape)
		cond = self.cond_proj(cond)
		# print(cond.shape, embed.shape)
		out = embed * cond

		out = self.emb_linear1(out)
		out = self.emb_linear2(out)
		return out

# 이거 이름을 mlpexplainer가 아니고 grad explainer로 바꿔야되나
class MLPExplainer(nn.Module):
    ''' Downstream MLP explainer based on gradient of output w.r.t. input embedding.
    Args:
        mlp_model: :obj:`torch.nn.Module` The downstream model to be explained.
        device: Torch CUDA device.
    '''

    def __init__(self, mlp_model, device):
        super(MLPExplainer, self).__init__()
        self.model = mlp_model.to(device)
        self.device = device

    def forward(self, embeds, mode='explain'):
        '''Returns probability by forward propagation or gradients by backward propagation
        based on the mode specified.
        '''
        embeds = embeds.detach().to(self.device)
        self.model.eval()
        if mode == 'explain':
            return self.get_grads(embeds)
        elif mode == 'pred':
            return self.get_probs(embeds)
        else:
            raise NotImplementedError

    def get_probs(self, embeds):
        logits = self.model(embeds)

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            probs = torch.cat([1-probs, probs], 1)
        else:
            probs = F.softmax(logits, dim=-1)
        return probs

    def get_grads(self, embeds):
        optimizer = torch.optim.SGD([embeds.requires_grad_()], lr=0.01)
        optimizer.zero_grad()
        logits = self.model(embeds)
        max_logits, _ = logits.max(dim=-1)
        max_logits.sum().backward()
        grads = embeds.grad
        grads = grads/torch.abs(grads).mean()
        return F.relu(grads)

class TAGExplainer(nn.Module):
    ''' The TAGExplainer that performs 2-stage explanations. Includes training and inference.
    Args:
        model: :obj:`torch.nn.Module`. the GNN embedding model to be explained.
        embed_dim: Integer. Dimension of node embeddings.
        device: Torch CUDA device.
        explain_graph: Boolean. Whether to explain a graph-level prediction task or 
            node-level prediction task.
        coff_size, coff_ent: Hyper-parameters for mask regularizations.
        grad_scale: Float. The scale parameter for generating random condition vectors.
        loss_type: String from "NCE" or "JSE". Type of the contrastive loss.
    '''
    def __init__(self, args, model, explain_graph: bool = True, loss_type = 'NCE'):
        """_
        

        args에 넣어야되는거:
            embed_dim, grad_scale, 
        """
        super(TAGExplainer, self).__init__()
        self.device = args.device
        self.embed_dim = args.embed_dim
        self.explain_graph = explain_graph
        self.model = model.to(self.device)

        self.explainer = ParamExplainer(self.embed_dim, explain_graph).to(self.device)

        # objective parameters for PGExplainer
        self.grad_scale = args.grad_scale
        self.coff_size = args.coff_size
        self.coff_ent = args.coff_ent
        self.t0 = args.coff_t0
        self.t1 = args.coff_te
        self.loss_type = loss_type

        # self._set_hops(num_hops) # 3
        # self.sampler = KHopSampler(self.num_hops)
          
        self.S = None


    def _set_hops(self, num_hops: int):
        """
        ㄱㅊ은듯
        나중에 써먹기
        """
        if num_hops is None:
            self.num_hops = sum(
                [isinstance(m, MessagePassing) for m in self.model.modules()])
        else:
            self.num_hops = num_hops


    def __set_masks__(self, edge_mask: Tensor):
        """ Set the edge weights before message passing
        Args:
            edge_mask (:obj:`torch.Tensor`): Edge weight matrix before message passing
              (default: :obj:`None`)
        """
        edge_mask = edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = edge_mask


    def __clear_masks__(self):
        """ clear the edge weights to None, and set the explain flag to :obj:`False` """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None


    def __loss__(self, embed: Tensor, pruned_embed: Tensor, 
        condition: Tensor, edge_mask: Tensor, **kwargs):
        '''
        embed: Tensor of shape [n_sample, embed_dim]
        pruned_embed: Tensor of shape [n_sample, embed_dim]
        condition: Tensor of shape [1, embed_dim]
        '''
        max_items = kwargs.get('max_items')
        if self.loss_type=='NCE':
            contrast_loss = NCE_loss([condition*embed, condition*pruned_embed])
        elif max_items and len(embed) > max_items:
            contrast_loss = self.__batched_JSE__(condition*embed, condition*pruned_embed, max_items)
        else:
            contrast_loss = JSE_loss([condition*embed, condition*pruned_embed])
        # print(condition.shape, embed, pruned_embed, contrast_loss)

        size_loss = self.coff_size * torch.mean(edge_mask)
        edge_mask = edge_mask * 0.99 + 0.005


        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent = self.coff_ent * torch.mean(mask_ent)

        loss = contrast_loss + size_loss + mask_ent
        return loss


    def __batched_JSE__(self, cond_embed, cond_pruned_embed, batch_size):
        loss = 0
        for i, (z1, z2) in enumerate(tDataLoader(
            TensorDataset(cond_embed, cond_pruned_embed), batch_size)):
            if len(z1)<=1:
                i -= 1
                break
            loss += JSE_loss([z1, z2])
        return loss/(i+1.0)

    def __rand_cond__(self, n_sample, max_val=None):
        lap = torch.distributions.laplace.Laplace(loc=0, scale=self.grad_scale)
        cond = F.relu(lap.sample([n_sample, self.embed_dim])).to(self.device)
        if max_val is not None:
            cond = torch.clip(cond, max=max_val)
        return cond

    def get_subgraph(self, node_idx: int, data):
        """prepare inductive하니까 필요없을듯
        """
        x, edge_index, edge_attr, y, batch = data.x, data.edge_index, data.edge_attr, data.y, data.batch
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        col, row = edge_index
        
        node_mask = self.sampler(edge_index, num_nodes, node_idx)
        edge_mask = node_mask[row] & node_mask[col]
        subset = torch.nonzero(node_mask).view(-1)
        edge_index, edge_attr = utils.subgraph(node_mask, edge_index, edge_attr, 
            relabel_nodes=True, num_nodes=num_nodes)

        x = x[subset]
        y = y[subset] if y is not None else None
        batch = batch[subset] if batch is not None else None

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch)
        return data, subset


    def concrete_sample(self, log_alpha: Tensor, beta: float = 1.0, training: bool = True):
        """ Sample from the instantiation of concrete distribution when training """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs
        else:
            gate_inputs = log_alpha

        return gate_inputs.sigmoid()


    # 이거는 외부에서 불릴일없는건가그러면
    def explain(self, graph, embed: Tensor, condition: Tensor,
                tmp: float = 1.0, training: bool = False, batch=False, **kwargs):
        """
        explain the GNN behavior for graph with explanation network
        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            embed (:obj:`torch.Tensor`): Node embedding matrix with shape :obj:`[num_nodes, dim_embedding]`
            tmp (:obj`float`): The temperature parameter fed to the sample procedure
            training (:obj:`bool`): Whether in training procedure or not
        Returns:
            probs (:obj:`torch.Tensor`): The classification probability for graph with edge mask
            edge_mask (:obj:`torch.Tensor`): The probability mask for graph edges
        """

        nodesize = embed.shape[0]
        feature_dim = embed.shape[1]

        x, adj, label = graph # label(int),
        
        coo_adj = coo_matrix(adj.detach().cpu())
        col, row = coo_adj.col, coo_adj.row,
        f1 = embed[col]
        f2 = embed[row]
        if self.explain_graph:
            out1 = global_max_pool(embed, batch)
            out2 = global_mean_pool(embed, batch)
            input_lin = torch.cat([out1, out2], dim=-1)
            f12self = input_lin#torch.cat([f1, f2], dim=-1)
            f12self = torch.cat([f1, f2], dim=-1)
        else:
            node_idx = kwargs.get('node_idx')
            self_embed = embed[node_idx].repeat(f1.shape[0], 1)
            f12self = torch.cat([f1, f2, self_embed], dim=-1)

        # using the node embedding to calculate the edge weight
        h = self.explainer(f12self.to(self.device), condition.to(self.device))
        mask_val = h.reshape(-1)
        values = self.concrete_sample(mask_val, beta=tmp, training=training)

        try:
            out_log = '%.4f, %.4f, %.4f, %.4f'%(
                h.max().item(), values.max().item(), h.min().item(), values.min().item())
        except:
            out_log = ''

        edge_mask = values
        '''
        mask_sparse = torch.sparse_coo_tensor(
            data.edge_index, values, (nodesize, nodesize)
        )
        # print((values>1).sum())
        mask_sigmoid = mask_sparse.to_dense()
        # print((mask_sigmoid>1).sum())

        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2

        edge_mask = sym_mask[col, row]
        '''

        # inverse the weights before sigmoid in MessagePassing Module
        # inv_sigmoid = lambda x: torch.log(x/(1-x))
        # 이거 대신에 adj에 씌워서 넣어야되지않나 masked_adj?
        # edge_mask를 nxn으로 바꾸거나 coo*edge_mask해서 nxn으로?
        print(edge_mask)
        exit()
        masked_adj = edge_mask * adj
        data.edge_weight = edge_mask


        # the model prediction with edge mask
        embed = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        data.edge_weight = None

        return embed, edge_mask, out_log


    def train_explainer_graph(self, loader, lr=0.001, epochs=10):
        """ training the explanation network by gradient descent(GD) using Adam optimizer """
        optimizer = Adam(self.explainer.parameters(), lr=lr)
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
    
    def explain_graph_func(self, loader, lr=0.001, epochs=10):
        optimizer = Adam(self.explainer.parameters(), lr=lr)
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
            try:
                pruned_embed, mask, log = self.explain(loader, embed=embed, 
                    condition=cond, tmp=tmp, training=True, batch=loader.batch.to(self.device))
            except:
                pruned_embed, mask, log = self.explain(loader, embed=embed, 
                    condition=cond, tmp=tmp, training=True, batch=torch.ones(loader.x.shape[0]).long().to(self.device))
            return pruned_embed, mask, log

                
    def train_large_explainer_node(self, loader, batch_size=2, lr=0.001, epochs=10, max_items=2000):
        """ training the explanation network by gradient descent(GD) using Adam optimizer """
        optimizer = Adam(self.explainer.parameters(), lr=lr, weight_decay=0.01)
            # train the mask generator
        for epoch in range(epochs):
            self.model.eval()
            self.explainer.train()
            for dt_idx, data in enumerate(loader):
                loss = 0.0
                optimizer.zero_grad()
                tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / epochs))
                data.to(self.device)
                
                with torch.no_grad():
                    try:
                        data = Batch.from_data_list([data])
                    except:
                        pass
                    try:
                        mask = data.train_mask
                    except:
                        mask = torch.ones_like(data.batch).bool()
                
                node_batches = torch.utils.data.DataLoader(torch.where(mask)[0].tolist(),
                    batch_size=batch_size, shuffle=True)
                pbar = tqdm(node_batches)
                for node_batch in pbar:
                    cond = self.__rand_cond__(1)
                    pruned_embeds, embeds = [], []
                    for node_idx in node_batch:
                        subgraph, subset = self.get_subgraph(node_idx=node_idx, data=data)
                        if subgraph.edge_index.shape[0]>10000 or subgraph.x.shape[0]>3000 or subgraph.x.shape[0]<2:
                            continue
                        new_node_idx = int(torch.where(subset == node_idx)[0])
                        with torch.no_grad():
                            subg_embeds = self.model(subgraph)
                        pruned_embed, mask, log = self.explain(subgraph, subg_embeds.to(self.device), 
                            condition=cond, tmp=tmp, training=True, node_idx=new_node_idx)
                        embeds.append(subg_embeds.cpu())#[new_node_idx:new_node_idx+1])
                        pruned_embeds.append(pruned_embed.cpu())#[new_node_idx:new_node_idx+1])
                    embeds = torch.cat(embeds, 0).to(self.device)
                    if len(embeds) <= 1:
                        continue
                    pruned_embeds = torch.cat(pruned_embeds, 0).to(self.device)
                    loss = self.__loss__(embeds, pruned_embeds, cond, mask)#, max_items=2000)
                    if torch.isnan(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.explainer.parameters(), 2.0)
                    optimizer.step()
                    pbar.set_postfix({'loss': loss.item(), 'log': log})
                

    def train_explainer_node(self, data, node_indice, batch_size=128, lr=0.001, epochs=10):
        """ training the explanation network by gradient descent(GD) using Adam optimizer """
        optimizer = Adam(self.explainer.parameters(), lr=lr)
            # train the mask generator
        for epoch in range(epochs):
            self.model.eval()
            self.explainer.train()

            loss = 0.0
            optimizer.zero_grad()
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / epochs))

            with torch.no_grad():
                all_embeds = self.model(data)

            self.model.to(self.device)
            data.to(self.device)

            # 700 길이의 False tensor 생성
            mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
            # 400부터 700까지 5 단위로 True로 설정
            mask[node_indice] = True

            node_batches = torch.utils.data.DataLoader(torch.where(mask)[0].tolist(),
                batch_size=batch_size, shuffle=True)

            pbar = tqdm(node_batches)

            for node_batch in pbar:

                cond = self.__rand_cond__(1) # 이 부분을 바꿔주면 downstream 적용 가능
                embeds = all_embeds[node_batch].to(self.device)
                pruned_embeds = []
                masks = []

                for node_idx in node_batch:
                    subgraph, subset = self.get_subgraph(node_idx=node_idx, data=data)
                    new_node_idx = int(torch.where(subset == node_idx)[0])

                    pruned_embed, mask, log = self.explain(subgraph, all_embeds.to(self.device)[subset],
                                                           condition=cond, tmp=tmp, training=True, node_idx=new_node_idx)

                    pruned_embeds.append(pruned_embed.cpu()[new_node_idx:new_node_idx+1])
                    masks.append(mask)
                pruned_embeds = torch.cat(pruned_embeds, 0).to(self.device)
                masks = torch.cat(masks, 0)
                if len(pruned_embeds)<=1:
                    continue

                loss = self.__loss__(embeds, pruned_embeds, cond, masks)

                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss.item(), 'log': log})


    def __edge_mask_to_node__(self, data, edge_mask, top_k):
        threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])

        hard_mask = (edge_mask > threshold).cpu()
        edge_idx_list = torch.where(hard_mask == 1)[0]
        selected_nodes = []
        edge_index = data.edge_index.cpu().numpy()
        for edge_idx in edge_idx_list:
            selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
        selected_nodes = list(set(selected_nodes))
        maskout_nodes = [node for node in range(data.x.shape[0]) if node not in selected_nodes]

        node_mask = torch.zeros(data.num_nodes).type(torch.float32).to(self.device)
        node_mask[maskout_nodes] = 1.0
        return node_mask, hard_mask


    # def forward(self, data, mlp_explainer:nn.Module, **kwargs):
    def forward(self, graph, mlp_explainer:nn.Module, node_idx, cond_vec, top_k=10):
        """ explain the GNN behavior for graph and calculate the metric values.
        The interface for the :class:`dig.evaluation.XCollector`.

        Args:
            data (tuple): (x, adj)
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            kwargs(:obj:`Dict`):
              The additional parameters
                - top_k (:obj:`int`): The number of edges in the final explanation results
                - y (:obj:`torch.Tensor`): The ground-truth labels

        :rtype: (:obj:`None`, List[torch.Tensor], List[Dict])
        """
        # top_k = kwargs.get('top_k') if kwargs.get('top_k') is not None else 10
        # node_idx = kwargs.get('node_idx')
        # cond_vec = kwargs.get('cond_vec')

        self.model.eval()
        mlp_explainer = mlp_explainer.to(self.device).eval()
        x, adj, label = graph # label(int),
        # data = data.to(self.device)

        if node_idx is not None:
            node_embed = self.model((x,adj)) # forward말고 embedding으로 해야되나
            embed = node_embed[node_idx:node_idx+1]
        elif self.explain_graph:
            embed, node_embed = self.model((x,adj))
        else:
            assert node_idx is not None, "please input the node_idx"

        probs = mlp_explainer(embed, mode='pred')
        grads = mlp_explainer(embed, mode='explain') if cond_vec is None else cond_vec
        probs = probs.squeeze()

        if self.explain_graph:
            subgraph = None
            target_class = torch.argmax(probs) if label is None else label
            # _, edge_mask, log = self.explain(data, embed=node_embed, condition=grads, tmp=1.0, training=False)
            ## sup
            _, edge_mask, log = self.explain(graph, embed=node_embed, condition=grads, tmp=1.0, training=False)
            
            ## unsup
            # _, edge_mask, log = self.explain(data, embed=node_embed, condition=torch.zeros_like(grads), tmp=1.0, training=False)
            node_mask = self.__edge_mask_to_node__(data, edge_mask, top_k)
            masked_data = mask_fn(data, node_mask)
            masked_embed = self.model(masked_data)
            masked_prob = mlp_explainer(masked_embed, mode='pred')
            masked_prob = masked_prob[:, target_class]
            sparsity_score = sum(node_mask) / data.num_nodes
        else:
            target_class = torch.argmax(probs)

            # target_class = torch.argmax(probs) if data.y is None else max(data.y[node_idx].long(), 0) # sometimes labels are +1/-1
            subgraph, subset = self.get_subgraph(node_idx=node_idx, data=data)
            new_node_idx = torch.where(subset == node_idx)[0]
            # _, edge_mask, log = self.explain(subgraph, node_embed[subset], condition=grads, 
            #     tmp=1.0, training=False, node_idx=new_node_idx, )
            
            _, edge_mask, log = self.explain(subgraph, node_embed[subset], condition=torch.zeros_like(grads), 
                tmp=1.0, training=False, node_idx=new_node_idx, )

            '''
                1. 여기서 distance 차이 구해서 내줘서 mask에 곱하기
                
                2. 그냥 처음부터 edge_weight에 곱하기
            '''

            # node_mask, hard_node_mask = self.__edge_mask_to_node__(subgraph, edge_mask, top_k)

            # masked_embed = self.model(mask_fn(subgraph, node_mask))
            # masked_prob = mlp_explainer(masked_embed, mode='pred')[new_node_idx, target_class]
            # masked_prob = mlp_explainer(masked_embed, mode='pred')[new_node_idx, target_class.long()]
            # sparsity_score = sum(node_mask) / subgraph.num_nodes

        # return variables
        # pred_mask = edge_mask.detach().cpu()
        # related_preds = [{
        #     'maskout': masked_prob.item(),
        #     'origin': probs[target_class].item(),
        #     'sparsity': sparsity_score}]
        return (subgraph, subset), edge_mask


class GradExplainer(Explainer):
    """"""