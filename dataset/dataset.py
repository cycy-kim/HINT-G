import torch

import pickle as pkl
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from utils import *

"""
가능하면 dataset 객체 싹다 하나로 ㄱㄱ
"""


# 이거 다시 확인해봐야할듯
NODE_SETTINGS = {
    """
    1: Original settings from GNNExplainer and PGExplainer
    2: Graphs containing motifs
    3: All graphs
    """

    'syn1': {
        1: [i for i in range(400,700,5)],
        # 2: 
    },

    'syn2': {},

    'syn3': {
        # 1: [i for i in range(511,871,6)],   # pos, neg unsup
        # 1: [i for i in range(511,871,24)],   # pos, neg unsup
        # 1: [i for i in range(511,871,6)],   # pos
        1: [i for i in range(0, 511, 9)],   # neg
        # 1: [i for i in range(450, 511, 9)],   # neg test
        # 1: [i for i in range(0, 511, 24)],   # neg test
        # 1: [i for i in range(733,734)],   # pos unsup vis
        2: [i for i in range(511,871)],
        # 3: [i for i in range(871)],
        3: [i for i in range(0,871,6)],
    },

    'syn4': {
        # 1: [i for i in range(511,800)], # motif있는것들싹다
        # 1: [i for i in range(511,800,6)], # pos neg
        1: [i for i in range(511,800, 24)], # pos neg
        # 1: [550, 610, 670, 730, 790], # 그냥 암거나
        2: [i for i in range(511,800)],
        # 3: [i for i in range(1231)],
        3: [i for i in range(0, 1231, 5)], # 너무많아서
    },
    
    'BA-2motif': {
        # 1: [i for i in range(0,100)] + [i for i in range(500,600)], # pos
        # 1: [i for i in range(505,506)], # vis
        1: [i for i in range(0,100)], # neg
        # 1: [i for i in range(0,100,5)], # neg test
        # 1: [44], # neg vis, iter20, scale100
        2: [i for i in range(1000)], # Each graph has its own motif.
        3: [i for i in range(1000)],
    },
}

class Extractor(object):
    def __init__(self, adj, features, edge_label_matrix, labels, hops, embeds=None):
        super(Extractor,self).__init__()
        self.adj = adj
        self.features = features
        self.embeds = embeds
        self.labels = labels
        self.hops = hops
        self.ext_adj = self.extend(adj,hops-1)
        self.edge_label_matrix = edge_label_matrix
        if isinstance(adj,np.ndarray):
            adj_coo = coo_matrix(adj)
        else:
            adj_coo = adj.tocoo()
        adj_list = []
        for _ in range(adj_coo.shape[0]):
            adj_list.append(set())

        for r,c in list(zip(adj_coo.row,adj_coo.col)):
            adj_list[r].add(c)
            adj_list[c].add(r)
        self.adj_list= adj_list

    def extend(self,adj,hops):
        ext_adj = adj.copy()
        for hop in range(hops):
            ext_adj = ext_adj @ adj + adj
        return ext_adj

    def subgraph(self,node):
        begin_index = self.ext_adj.indptr[node]
        end_index = self.ext_adj.indptr[node+1]
        subnodes = set(self.ext_adj.indices[begin_index:end_index])
        subnodes.add(node)
        remap = {}
        remap[node] = 0
        nodes = [node]
        for n in subnodes:
            if n not in remap:
                remap[n]=len(remap)
                nodes.append(n)
        row = []
        col = []
        data = []
        edge_label = []
        for n in remap:
            newid = remap[n]
            for nb in self.adj_list[n]:
                if nb in remap:
                    nb_new_id = remap[nb]
                    row.append(newid)
                    col.append(nb_new_id)
                    data.append(1.0)
                    edge_label.append(self.edge_label_matrix[n,nb])
        sub_adj = coo_matrix((data,(row,col)),shape=(len(remap),len(remap)))
        sub_edge_label_matrix = coo_matrix((edge_label,(row,col)),shape=(len(remap),len(remap)))

        sub_features = self.features[nodes]
        sub_labels = self.labels[nodes]
        if self.embeds is not None:
            sub_embeds = self.embeds[nodes]
        else:
            sub_embeds = None
        return sub_adj, sub_features, sub_labels, sub_edge_label_matrix, sub_embeds

class Dataset():
    """"""

class SyntheticDataset(Dataset):
    def __init__(self, args, embeds=None):
        """
        dataset: syn1~syn4, BA-2motif
        """
        self.args = args

        self.dataset = args.dataset
        self.nodes = NODE_SETTINGS[args.dataset][args.setting]
        self.sub_support_tensors = None
        self.sub_label_tensors = None
        self.sub_features = None
        self.sub_embeds = None
        self.sub_adjs = None
        self.sub_edge_labels = None
        self.sub_labels = None
        self.remap = None

        self._load_dataset()
        

    def _load_dataset(self):
        if self.dataset == "BA-2motif":
            with open(f'dataset/BA-2motif.pkl','rb') as fin:
                (self.adjs, self.feas, self.labels) = pkl.load(fin)
        else:
            with open(f'dataset/{self.dataset}.pkl', 'rb') as fin:
                adj, self.features, \
                self.y_train, self.y_val, self.y_test, \
                self.train_mask, self.val_mask, self.test_mask, \
                self.edge_label_matrix = pkl.load(fin)

            self.adj = csr_matrix(adj)
            self.features_tensor = torch.tensor(self.features, dtype=torch.float32)
            self.train_mask_tensor = torch.tensor(self.train_mask, dtype=torch.bool)
            self.all_label = np.logical_or(self.y_train, np.logical_or(self.y_val, self.y_test))
            
            support = self._preprocess_adj(self.adj)
            support_indices = torch.tensor(support[0], dtype=torch.int64)
            support_values = torch.tensor(support[1])
            support_shape = tuple(support[2])

            self.support_tensor = torch.sparse_coo_tensor(support_indices.t(), support_values, support_shape)
            # 안헷갈리게 마지막에 self. = 로 정리하기

    def cuda(self):
        device = self.args.device
        def move_to_device(item):
            if isinstance(item, torch.Tensor):
                return item.to(device)
            elif isinstance(item, list):
                return [move_to_device(i) for i in item]
            else:
                return item
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                setattr(self, attr_name, move_to_device(attr))
        
    def prepare_inductive(self, embeds=None):
        assert self.dataset != "BA-2motif", "The BA-2motif dataset does not support inductive setting."
    
        # train_mask_tensor = torch.tensor(self.train_mask, dtype=torch.bool)

        # support = self._preprocess_adj(adj_csr)
        # features_tensor = torch.tensor(self.features, dtype=torch.float32)
        # support_indices = torch.tensor(support[0], dtype=torch.int64)
        # support_values = torch.tensor(support[1])
        # support_shape = tuple(support[2])
        # support_tensor = torch.sparse.FloatTensor(support_indices.t(), support_values, support_shape)

        hops = len(self.args.hiddens.split('-'))
        if self.args.dataset=='syn4':
            hops += 1
        extractor = Extractor(self.adj, self.features, self.edge_label_matrix, self.all_label, hops, embeds)

        sub_support_tensors = []
        sub_label_tensors = []
        sub_features = []
        sub_embeds = []
        sub_adjs = []
        sub_edge_labels = []
        sub_labels = []
        remap = {}

        for node in self.nodes:
            sub_adj, sub_feature, sub_label, sub_edge_label_matrix, sub_embed = extractor.subgraph(node)
            remap[node] = len(sub_adjs)
            sub_support = self._preprocess_adj(sub_adj)
            
            sub_support_tensor = torch.sparse_coo_tensor(
                torch.LongTensor(sub_support[0].T), 
                torch.FloatTensor(sub_support[1]), 
                torch.Size(sub_support[2])
            )
            sub_label_tensor = torch.tensor(sub_label, dtype=torch.float32)

            sub_adjs.append(sub_adj)
            sub_features.append(torch.tensor(sub_feature).float())
            if sub_embed is not None:
                sub_embeds.append(torch.tensor(sub_embed).float())
            sub_labels.append(sub_label)
            sub_edge_labels.append(sub_edge_label_matrix)
            sub_label_tensors.append(sub_label_tensor)
            sub_support_tensors.append(sub_support_tensor)

        # if self.args.dataset == 'syn3':
        #     with open('dataset/syn3_node_labels', 'rb') as file:
        #         sub_labels = pkl.load(file)

        self.sub_support_tensors = sub_support_tensors
        self.sub_label_tensors = sub_label_tensors
        self.sub_features = sub_features
        self.sub_embeds = sub_embeds
        self.sub_adjs = sub_adjs
        self.sub_edge_labels = sub_edge_labels
        self.sub_labels = sub_labels
        self.remap = remap


    def pad_graph(self, graph, max_num_nodes):
        """
        Pads a graph to have a uniform number of nodes.

        Args:
            graph ((features, support, labels)): 
                features: [B, num_node, feat_dim] (dense tensor)
                support: [B, num_node, num_node] (dense tensor)
                labels: [B, num_node]
            max_num_nodes (int): Number of nodes to pad to.

        Returns:
            (padded_features, padded_support, padded_labels): 
                padded_features: [B, max_num_nodes, feat_dim]
                padded_support: [B, max_num_nodes, max_num_nodes]
                padded_labels: [B, max_num_nodes]
        """
        features, support, labels = graph
        B, num_node, feat_dim = features.size()

        # Padding features with mean of existing features
        mean_features = features.mean(dim=1, keepdim=True)  # [B, 1, feat_dim]
        padded_features = torch.zeros((B, max_num_nodes, feat_dim), device=features.device)
        padded_features[:, :num_node, :] = features
        padded_features[:, num_node:, :] = mean_features

        # Convert support to dense if it is sparse
        if support.is_sparse:
            support = support.to_dense()

        # Padding support (adjacency matrix) with zeros
        padded_support = torch.zeros((B, max_num_nodes, max_num_nodes), device=support.device)
        padded_support[:, :num_node, :num_node] = support

        # Padding labels with zeros
        padded_labels = torch.zeros((B, max_num_nodes), dtype=labels.dtype, device=labels.device)
        padded_labels[:, :num_node] = labels

        return padded_features, padded_support, padded_labels


    def _preprocess_adj(self, adj, norm=False):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        if norm:
            adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
            return sparse_to_tuple(adj_normalized)
        else:
            return sparse_to_tuple(sp.coo_matrix(adj))

class MutagDataset(Dataset):
    def __init__(self, args, embeds=None):
        """
        dataset: syn1~syn4, BA-2motif
        """
        self.args = args

        self.dataset = args.dataset
        self.nodes = None

        self._load_dataset()

    def _get_graph_data(self):
        dataset = args.dataset
        pri = 'dataset/' + dataset + '/' + dataset + '_'

        file_edges = pri+'A.txt'
        # file_edge_labels = pri+'edge_labels.txt'
        file_edge_labels = pri+'edge_gt.txt'
        file_graph_indicator = pri+'graph_indicator.txt'
        file_graph_labels = pri+'graph_labels.txt'
        file_node_labels = pri+'node_labels.txt'

        edges = np.loadtxt( file_edges,delimiter=',').astype(np.int32)
        try:
            edge_labels = np.loadtxt(file_edge_labels,delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

        graph_indicator = np.loadtxt(file_graph_indicator,delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(file_graph_labels,delimiter=',').astype(np.int32)

        try:
            node_labels = np.loadtxt(file_node_labels,delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use node label 0')
            node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

        graph_id = 1
        starts = [1]
        node2graph = {}
        for i in range(len(graph_indicator)):
            if graph_indicator[i]!=graph_id:
                graph_id = graph_indicator[i]
                starts.append(i+1)
            node2graph[i+1]=len(starts)-1
        # print(starts)
        # print(node2graph)
        graphid  = 0
        edge_lists = []
        edge_label_lists = []
        edge_list = []
        edge_label_list = []
        for (s,t),l in list(zip(edges,edge_labels)):
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid!=tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s,t,'graph id',sgid,tgid)
                exit(1)
            gid = sgid
            if gid !=  graphid:
                edge_lists.append(edge_list)
                edge_label_lists.append(edge_label_list)
                edge_list = []
                edge_label_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append((s-start,t-start))
            edge_label_list.append(l)

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)

        # node labels
        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(node_labels)):
            nid = i+1
            gid = node2graph[nid]
            # start = starts[gid]
            if gid!=graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(node_labels[i])
        node_label_lists.append(node_label_list)

        return edge_lists, graph_labels, edge_label_lists, node_label_lists


    def _load_dataset(self):
        """
        !!Graph Classification!!
        Only consider the mutagen graphs with NO2 and NH2.
        """
        if self.dataset != "Mutagenicity":
            raise NameError('MutagDataset only supports Mutagenicity only')
        
        edge_lists, graph_labels, edge_label_lists, node_label_lists = self._get_graph_data()
        with open('dataset/Mutagenicity.pkl','rb') as fin:
            original_adjs, original_features, original_labels = pkl.load(fin)
                

        # train_mask_tensor
        selected =  []
        for gid in range(original_adjs.shape[0]):
            if np.argmax(original_labels[gid]) == 0 and np.sum(edge_label_lists[gid]) > 0:
                selected.append(gid)
            # selected.append(gid)
        print('number of mutagen graphs with NO2 and NH2',len(selected))
        
        self.adjs = original_adjs[selected]
        self.feas = original_features[selected]
        self.labels = original_labels[selected]
        self.edge_lists = [edge_lists[i] for i in selected]
        self.edge_label_lists=[edge_label_lists[i] for i in selected]
        self.nodes = list(range(self.adjs.shape[0]))

    def cuda(self):
        device = self.args.device
        def move_to_device(item):
            if isinstance(item, torch.Tensor):
                return item.to(device)
            elif isinstance(item, list):
                return [move_to_device(i) for i in item]
            else:
                return item
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                setattr(self, attr_name, move_to_device(attr))

