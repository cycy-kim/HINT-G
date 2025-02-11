import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import k_hop_subgraph

import copy
from tqdm import tqdm

import numpy as np
from sklearn.metrics import roc_auc_score

from models import GCN_Node as GCN
from utils import args

@torch.no_grad()
def test(model, data, args):
    model.eval()

    x = data.x
    num_nodes = x.shape[0]
    edge_index = data.edge_index  # (2, num_edges)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=args.device)
    adj[edge_index[0], edge_index[1]] = 1

    out = model((x,adj))
    test_mask = data.test_mask

    test_labels = data.y[test_mask]
    pred = out[test_mask].max(1)[1]
    # print(out.shape)
    probabilities = F.softmax(out[test_mask], dim=1)
    # print(test_labels.shape)
    # print(probabilities.shape)
    roc_auc = roc_auc_score(test_labels.cpu(), probabilities.cpu().numpy(), multi_class="ovo")
    accuracy = pred.eq(test_labels).sum().item() / test_mask.sum().item()
    # print("accuracy: ", accuracy)
    # print("roc_auc: ", roc_auc)
    # print('-----------------')
    return roc_auc, accuracy

@torch.no_grad()
def test_inductive(model, subgraphs, args):
    model.eval()
    accuracies = []

    for subgraph in subgraphs:
        x = subgraph['x'].to(args.device)
        edge_index = subgraph['edge_index'].to(args.device)
        target_label = subgraph['y'][0].to(args.device)

        num_nodes = x.shape[0]
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=args.device)
        adj[edge_index[0], edge_index[1]] = 1

        out = model((x, adj))
        pred = out[0].max(0)[1]

        accuracy = pred.eq(target_label).item()
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)

    return -1, mean_accuracy

def train(model, data, args):
    # model = GCN(dataset.num_node_features, dataset.num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    model.train()
    x = data.x
    num_nodes = x.shape[0]
    edge_index = data.edge_index  # (2, num_edges)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=args.device)
    adj[edge_index[0], edge_index[1]] = 1
    
    best_accuracy = -np.inf
    for epoch in tqdm(range(1000)):
        model.train()
        optimizer.zero_grad()
        out = model((x,adj))
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        rocauc, accuracy = test(model, data, args)
        if best_accuracy <= accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.model_weight_path)
            print(f'model saved to {args.model_weight_path} with rocauc {rocauc}, acc {accuracy}')
        # else:
            # print(f'accuracy : {accuracy}')


def find_k_hops(adj, nodes, hops):
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

def hvps(grad_all, model_params, h_estimate):
    element_product = 0
    for grad_elem, v_elem in zip(grad_all, h_estimate):
        element_product += torch.sum(grad_elem * v_elem)
    
    return_grads = grad(element_product,model_params,create_graph=True)
    return return_grads
           
def gif_approxi(model, grads, nodeid, losses):
    # 이거 args에서?
    iteration = 3
    scale = 100
    damp = 0
    grad_all, grad1, grad2 = grads
    loss, loss1, loss2 = losses
    original_params = [p.data for p in model.parameters() if p.requires_grad]
    v = tuple(grad1_t - grad2_t for grad1_t, grad2_t in zip(grad1,grad2))
    h_estimate = tuple(grad1_t - grad2_t for grad1_t, grad2_t in zip(grad1,grad2))
    for _ in range(iteration):
        model_params  = [p for p in model.parameters() if p.requires_grad]
        hv            = hvps(grad_all, model_params, h_estimate)
        with torch.no_grad():
            h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                        for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]
    
    params_change = [h_est * v1 for v1, h_est in zip(v, h_estimate)]

    score = torch.mean(torch.stack([torch.mean(torch.abs(t)/torch.abs(w)) for w, t in zip(original_params, params_change)]))
    return score

def pos_singlegraph(model, graph, whole_graph):
    score_edges = []

    x = graph['x']
    edge_index = graph['edge_index']
    label = graph['y']

    whole_x = whole_graph['x']
    whole_edge_index = whole_graph['edge_index']
    whole_label = whole_graph['y']
    whole_num_nodes = whole_x.shape[0]
    whole_adj = torch.zeros((whole_num_nodes, whole_num_nodes), dtype=torch.float32, device=args.device)
    whole_adj[whole_edge_index[0], whole_edge_index[1]] = 1

    num_nodes = x.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=args.device)
    adj[edge_index[0], edge_index[1]] = 1
    optimizer = optim.Adam(model.parameters())
    for e_i in range(edge_index.shape[-1]):
        optimizer.zero_grad() # just to save gpu memory인데 딱히 효과 없는듯
        src, des = edge_index[0][e_i].item(), edge_index[1][e_i].item()
        if src >= des:
            continue

        new_adj = copy.deepcopy(adj)
        new_adj[src][des] = False
        new_adj[des][src] = False

        original_output = model((x, adj))
        perturbed_output = model((x, new_adj))

        row_indices, col_indices = new_adj.nonzero(as_tuple=True)

        row_indices = row_indices.long()
        col_indices = col_indices.long()

        indices = torch.stack([row_indices, col_indices])
        values = new_adj[row_indices, col_indices]  # Extract values for non-zero entries
        values = values.float()  # Ensure float type

        # Create sparse tensor
        perturbed_support_tensor = torch.sparse_coo_tensor(indices, values, new_adj.size())

        influenced_nodes = find_k_hops(perturbed_support_tensor, [src, des], 3)
        mask1 = np.array([False] * perturbed_output.shape[0])
        mask1[influenced_nodes] = True
        mask2 = mask1


        original_whole_output = model((whole_x, whole_adj))
        all_label_tensor = whole_label

        loss = F.nll_loss(original_whole_output[whole_graph.train_mask], all_label_tensor[whole_graph.train_mask])
        loss1 = F.nll_loss(original_output[mask1], label[mask1])
        loss2 = F.nll_loss(perturbed_output[mask2], label[mask2])
        model_params = [p for p in model.parameters() if p.requires_grad]
        
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

        score = gif_approxi(model, (grad_all, grad1, grad2), 0, (loss, loss1, loss2))
        score_edges.append((score.item(), (src, des)))


    score_edges.sort(reverse=True, key=lambda x: x[0])
    return score_edges

# @torch.no_grad()
# def pred_changes(model, subgraph, remaining_edges):

#     x = subgraph['x']
#     edge_index = subgraph['edge_index']
#     label = subgraph['y']

#     num_nodes = x.shape[0]
#     adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=args.device)
#     adj[edge_index[0], edge_index[1]] = 1

#     original_output = model((x, adj))
#     original_pred = torch.argmax(original_output[0], dim=-1).item()
#     for e_i in range(edge_index.shape[-1]):
#         src, des = edge_index[0][e_i].item(), edge_index[1][e_i].item()
#         if src >= des:
#             continue
        
#         new_adj = copy.deepcopy(adj)
#         if (src,des) not in remaining_edges:
#             new_adj[src][des] = False
#             new_adj[des][src] = False
#     new_output = model((x,new_adj))
#     new_pred = torch.argmax(new_output[0], dim=-1).item()
    
#     return not(original_pred == new_pred)


# 이거는 bool 말고 probability 변화량 반환
@torch.no_grad()
def pred_changes(model, subgraph, remaining_edges):

    x = subgraph['x']
    edge_index = subgraph['edge_index']
    label = subgraph['y'][0] # target nodeid 0
    # exit()
    num_nodes = x.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=args.device)
    adj[edge_index[0], edge_index[1]] = 1

    original_output = torch.softmax(model((x, adj)), dim=-1)
    # original_pred = torch.argmax(original_output[0], dim=-1).item()
    for e_i in range(edge_index.shape[-1]):
        src, des = edge_index[0][e_i].item(), edge_index[1][e_i].item()
        if src >= des:
            continue
        
        new_adj = copy.deepcopy(adj)
        # if (src,des) not in remaining_edges: # top 30% edge만 남김
        #     new_adj[src][des] = False
        #     new_adj[des][src] = False
        if (src,des) in remaining_edges: # top 30% edge를 제거
            new_adj[src][des] = False
            new_adj[des][src] = False
    new_output = torch.softmax(model((x,new_adj)), dim=-1)
    # new_pred = torch.argmax(new_output[0], dim=-1).item()
    # print(original_output[0][label])
    # print(new_output[0][label])
    # exit()
    return (original_output[0][label] - new_output[0][label]).item() # target nodeid 0

def edge_influences(model, subgraphs, data):
    """
    Returns Fidelity

    Args:
        model (_type_): _description_
        subgraphs (_type_): _description_
    """

    fidelities = []
    train_mask = data.train_mask
    for graph_idx, subgraph in enumerate(tqdm(subgraphs)):
        if not train_mask[graph_idx]: # 학습때쓴걸로만 explain
            continue
        score_edges = pos_singlegraph(model, subgraph, data)
        # fidelities.append(fidelity)
        # print(np.ceil(len(score_edges)*0.3))
        remaining_edges = [edge for _,edge in score_edges[:int(len(score_edges)*0.3)]]
        pred_change = pred_changes(model, subgraph, remaining_edges)
        # print(pred_changed)

        fidelities.append(pred_change)
        # if pred_changed: 
        #     fidelities.append(0)
        # else:   # 안바뀔수록 fidelity up
        #     fidelities.append(1)

        print(f'{np.sum(fidelities)}/{len(fidelities)}')
    return np.mean(fidelities)

def extract_subgraphs(data, k=3):
    subgraphs = []
    for node in range(data.num_nodes):
        # 기준 노드에서 k-hop 서브그래프 추출
        node_idx, edge_index, _, _ = k_hop_subgraph(
            node_idx=node,
            num_hops=k,
            edge_index=data.edge_index,
            relabel_nodes=True
        )
        
        # 서브그래프 노드 특징과 라벨 추출
        subgraph_x = data.x[node_idx]
        subgraph_y = data.y[node_idx]
        
        # 새로운 서브그래프 데이터 구성
        subgraph = {
            'edge_index': edge_index,
            'x': subgraph_x,
            'y': subgraph_y,
            'target_node': 0  # 중심 노드는 항상 0번으로 설정
        }
        subgraphs.append(subgraph)
    return subgraphs

if __name__=="__main__":
    """"""
    args.dataset='cora'
    args.device='cuda'
    args.gnn_type='supervised' # 'supervised' or 'unsupervised'
    args.task = 'pos' # 'pos' or 'neg'

    args.lr = 0.001
    args.hiddens='512-512-512'




    args.gnn_task = 'node' if args.dataset[:3] == 'syn' else 'graph'
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type
    
    data = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())[0].to(args.device)
    cora_num_labels = 7
    subgraphs = extract_subgraphs(data,k=3)
    # print(len(subgraphs))
    # print(subgraphs[0])
    # exit()
    model = GCN(args=args, input_dim=data.x.shape[-1], output_dim=cora_num_labels).to(args.device)
    model.load_state_dict(torch.load(args.model_weight_path, weights_only=True))

    # train(model, data, args)
    # print(test(model,data,args))
    print(test_inductive(model,subgraphs, args))

    fidelity = edge_influences(model, subgraphs, data)
    print(f'fidelity : {fidelity}')




# print(data.x)
# print(data.edge_index)
# print(data.y)