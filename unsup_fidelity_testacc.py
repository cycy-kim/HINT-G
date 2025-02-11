import time
import random
import math

import torch

import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from models import XAIFG, XAIFG_Node, Explainer
from utils import args
from dataset.dataset import NODE_SETTINGS

import warnings
warnings.simplefilter("ignore", category=UserWarning)

''' TODO Implementation
각 subgraph에서 logistic regression한번씩 해서 accuracy반환하는게 아니고

1. (511, 800)을 train test로 split해서

2. train으로만 logistic regression 학습시키는데 여기서
-random edge k개 추가해서 train시켜보고
-homo edge k개
-nonhomo edge k개(homo보다는 nonhomo 추가하는게 방해가 된다)(X)
-neg edge k개 추가(nonhomo를 xaifg가 잘or더 잘 찾는다(unsup이라 label을 모르니까..))

3. test set기준
- 그냥 accuracy 반환
'''

''' TODO
1. topk말고 %마다 결과 어떻게 나오는지
2. rand non-homo edge를 넣은게 rand중에서는 제일 낮게....

'''

@torch.enable_grad()
def get_topk_neg_edges(explainer:Explainer, nodeid, topk=1):
    """_summary_

    Args:
        explainer (Explainer): XAIFG model
        nodeid (Int): NOT remapped, original nodeid

    Returns:
        topk_neg_edges (List of tuple): [(src1, des1), (src2, des2), ... ], src <= des
    """

    score_edges = explainer._neg_singlegraph(nodeid)
    edges = [edge for score, edge in score_edges]
    topk_neg_edges = edges[:topk] # score_edges is sorted
    
    return topk_neg_edges

def get_sub_label(explainer:Explainer, nodeid):
    """
    sub_labels는 안정확하고
    sub_edge_labels는 정확해서
    sub_edge_labels로 sub_labels 계산하기

    Args:
        explainer (Explainer): XAIFG model
        nodeid (_type_): _description_
    """
    
    data = explainer.data
    sub_edge_label = data.sub_edge_labels[data.remap[nodeid]].todense()
    num_node = len(sub_edge_label)
    
    sub_label = torch.zeros(num_node)

    for src in range(num_node):
        for des in range(num_node):
            if sub_edge_label[src,des] or sub_edge_label[des,src]:
                sub_label[src] = True
                sub_label[des] = True
    
    
    return sub_label

@torch.no_grad()
def get_trained_classifier(explainer, train_nodes, add_cond=None, topk=1):
    """_summary_

    Args:
        explainer (_type_): _description_
        train_nodes (_type_): _description_
        add_cond (str): 'rand-homo', 'rand-non-homo', 'rand', 'neg'. Defaults to None.
        topk (int or float): add_cond가 None이 아닐때만 사용
                        int: 몇개넣을지.. rand면 top아니고 그냥 k로 쓰임
                        float: 해당 subgraph에서 넣을수 있는 edge중 몇퍼센트를 넣을지..0.5=50%, 100%일때는 1이 아니고 1.0
    Returns:
        _type_: trained classifier
    """
    X = []
    y = []
    data = explainer.data
    
    for nodeid in train_nodes:
        sub_feature_tensor = data.sub_features[data.remap[nodeid]]
        sub_support_tensor = data.sub_support_tensors[data.remap[nodeid]]

        new_indices = sub_support_tensor.coalesce().indices().clone()
        new_values = sub_support_tensor.coalesce().values().clone()
        size = sub_support_tensor.coalesce().size()

        num_nodes = size[0]
        if num_nodes == 9:
            continue

        sub_label = get_sub_label(explainer, nodeid)
        existing_edges = set()
        cond_edges = set()
        indices_np = new_indices.cpu().numpy()
        for src, des in zip(indices_np[0], indices_np[1]):
            if src == des:
                continue
            edge = (min(src, des), max(src, des))
            existing_edges.add(edge)


        all_possible_edges = {(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)}
        if add_cond == 'rand':
            for src in range(num_nodes):
                for des in range(num_nodes):
                    """do nothing"""

        elif add_cond == 'rand-homo':
            for src in range(num_nodes):
                for des in range(num_nodes):
                    if des >= src:
                        continue
                    if sub_label[src] != sub_label[des]: # non homophily edges를 제거
                        cond_edges.add((src,des))
                    
        elif add_cond == 'rand-non-homo':
            for src in range(num_nodes):
                for des in range(num_nodes):
                    if des >= src:
                        continue
                    if sub_label[src] == sub_label[des]: # homophily edges를 제거
                        cond_edges.add((src,des))
        
        
        missing_edges = list(all_possible_edges - existing_edges - cond_edges) # missing edges에서 sampling

        if isinstance(topk, float):
            cur_topk = math.ceil(len(missing_edges)*topk)
        elif isinstance(topk, int):
            cur_topk = topk
            

        if add_cond is None:
            edges_to_add = []
        elif add_cond[:4] == 'rand':
            if len(missing_edges) < cur_topk:
                edges_to_add = missing_edges
            else:
                edges_to_add = random.sample(missing_edges, cur_topk)
        elif add_cond is not None:
            edges_to_add = get_topk_neg_edges(explainer, nodeid, cur_topk)
        
        for (src, des) in edges_to_add:
            new_edge1 = torch.tensor([[src], [des]], device=new_indices.device)
            new_edge2 = torch.tensor([[des], [src]], device=new_indices.device)
            new_value = torch.tensor([1.0], device=new_values.device)

            new_indices = torch.cat([new_indices, new_edge1, new_edge2], dim=1)
            new_values = torch.cat([new_values, new_value, new_value])

        added_sub_support_tensor = torch.sparse_coo_tensor(new_indices, new_values, size)

        out = explainer.model(sub_feature_tensor, added_sub_support_tensor)
        sub_label = get_sub_label(explainer, nodeid)

        X.extend(out.cpu().numpy())
        y.extend(sub_label.numpy())

    clf = LogisticRegression(random_state=42, max_iter=3000)
    clf.fit(X, y)
    return clf

@torch.no_grad()
def test_accuracy(explainer, clf, test_nodes):
    X_test = []
    y_test = []
    data = explainer.data

    for nodeid in test_nodes:
        sub_feature_tensor = data.sub_features[data.remap[nodeid]]
        sub_support_tensor = data.sub_support_tensors[data.remap[nodeid]]

        out = explainer.model(sub_feature_tensor, sub_support_tensor)
        sub_label = get_sub_label(explainer, nodeid)

        X_test.extend(out.cpu().numpy())
        y_test.extend(sub_label.numpy())

    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    return acc

def neg_fidelity(explainer, topk, train_ratio):
    # whole_nodes = [i for i in range(511, 800)]
    whole_nodes = NODE_SETTINGS[args.dataset][args.setting]

    split_idx = int(len(whole_nodes) * train_ratio)
    random.shuffle(whole_nodes)
    train_nodes, test_nodes = whole_nodes[:split_idx], whole_nodes[split_idx:]

    original_clf = get_trained_classifier(explainer, train_nodes, add_cond=None)
    original_acc = test_accuracy(explainer, original_clf, test_nodes)

    neg_added_clf = get_trained_classifier(explainer, train_nodes, add_cond='neg', topk=topk)
    neg_added_acc = test_accuracy(explainer, neg_added_clf, test_nodes)
    
    rand_iter = 10
    rand_accs = []
    homo_accs = []
    nonhomo_accs = []
    
    for _ in tqdm(range(rand_iter)):
        rand_added_clf = get_trained_classifier(explainer, train_nodes, add_cond='rand', topk=topk)
        rand_homo_added_clf = get_trained_classifier(explainer, train_nodes, add_cond='rand-homo', topk=topk)
        rand_non_homo_added_clf = get_trained_classifier(explainer, train_nodes, add_cond='rand-non-homo', topk=topk)
        
        rand_added_acc = test_accuracy(explainer, rand_added_clf, test_nodes)
        rand_homo_added_acc = test_accuracy(explainer, rand_homo_added_clf, test_nodes)
        rand_non_homo_added_acc = test_accuracy(explainer, rand_non_homo_added_clf, test_nodes)
        
        rand_accs.append(rand_added_acc)
        homo_accs.append(rand_homo_added_acc)
        nonhomo_accs.append(rand_non_homo_added_acc)

    print(f'original, accuracy : {original_acc}')
    print(f'{topk} neg edges found by XAIFG added, mean accuracy : {neg_added_acc}')
    
    print(f'{topk} random edges added, mean accuracy : {np.mean(rand_accs)}')
    print(f'{topk} random homo edges added, mean accuracy : {np.mean(homo_accs)}')
    print(f'{topk} random non-homo edges added, mean accuracy : {np.mean(nonhomo_accs)}')

def neg_fidelity_grid(explainer):
    topks = [1,2,3,4,5,6,7]
    train_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    for train_ratio in tqdm(train_ratios, desc="Processing Train Ratios"):
        for topk in tqdm(topks, desc=f"Train Ratio {train_ratio}", leave=False):
            neg_fidelity(explainer, topk, train_ratio)
        
if __name__== '__main__':
    args.device = 'cuda'

    # node
    args.dataset = 'syn4' # syn3는 추가됐을때 class가 실제로 바뀌니까.. 안바뀌는 grid로
    
    args.gnn_type = 'unsupervised' # 'supervised' or 'unsupervised'
    args.hidden_dim = 128 # unsupervised일때만
    args.task = 'neg' # 'pos' or 'neg'
    args.setting = 1 # neg일때 setting 잘보고하기
    
    args.iter = 3
    args.scale = 100000

    ###
    # args.topk = 1
    # args.train_ratio = 0.1
    ###

    ######

    args.gnn_task = 'node' if args.dataset[:3] == 'syn' else 'graph'

    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type

    explainer = XAIFG_Node(args=args)
    # explainer = XAIFG(args=args)
    explainer.model.eval()

    # rocauc, acc, precision, recall = explainer.edge_influences()
    # print(rocauc)

    # neg_fidelity_grid(explainer)
    # topk개 말고 %로 하는것도 ㄱㄱ subgraph마다 complete-existing
    neg_fidelity(explainer, topk=args.topk, train_ratio=args.train_ratio)

