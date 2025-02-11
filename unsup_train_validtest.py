import os
import glob

import torch
import torch.nn.functional as F
import torch_geometric.utils as torch_geometric_utils

import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression

# stats
# import dcor
from scipy.stats import spearmanr, kendalltau
from scipy.stats import kurtosis, skew, entropy
from sklearn.feature_selection import mutual_info_regression
# from minepy import MINE


from models import GCN_Emb
from dataset import Data, LinkNeighborLoader
from models import XAIFG as XAIFG_Edge, XAIFG_Node
from utils import distance_correlation, args
from utils import *

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
args.epochs = 3000 #500
args.lr = 0.001
args.weight_decay= 3e-4 #5e-4

args.dataset='syn4'
args.hidden_dim = 128
# args.num_layers = 5 # 이거 explainer prepare_model에서 직접바꿔야됨
    
model_weight_path = args.save_path +'_valid_test/' + args.dataset + '_unsupervised_torch_testing'

    
with open('./dataset/' + args.dataset + '.pkl', 'rb') as fin:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pkl.load(fin)



        
features = torch.FloatTensor(features)
y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test

y = torch.tensor(y).argmax(dim=-1)
# y = torch.tensor(y_train + y_val + y_test).argmax(dim=-1)

train_mask = torch.where(torch.tensor(train_mask))[0]
val_mask = torch.where(torch.tensor(val_mask))[0]
test_mask = torch.where(torch.tensor(test_mask))[0]

edge_index, _ = torch_geometric_utils.dense_to_sparse(torch.tensor(adj))
edge_index = edge_index
edge_label, _ = torch_geometric_utils.dense_to_sparse(torch.tensor(edge_label_matrix))


data = Data(x=features, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask


# train_data = data.subgraph(data.train_mask)
train_loader = LinkNeighborLoader(data, batch_size=1024, shuffle=True,
                                neg_sampling_ratio=1.0,num_neighbors=[10,10])
# train_loader = LinkNeighborLoader(train_data, batch_size=128, shuffle=True,
#                                 neg_sampling_ratio=1.0,num_neighbors=[10,10])
# train_loader = LinkNeighborLoader(data, batch_size=128, shuffle=True,
#                                 neg_sampling_ratio=1.0,num_neighbors=[10,10])

# @torch.no_grad()
# def test(out):
#     X = out.cpu().detach().numpy()
#     y = data.y.cpu().detach().numpy()
#     # X = out
#     # node classification
    
#     # clf = LogisticRegression(random_state=42, max_iter=300).fit(X_train, y_train)
#     # score = clf.score(X_test, y_test) # accuracy

#     train_mask = data.train_mask.cpu().detach().numpy()
#     test_mask = data.test_mask.cpu().detach().numpy()
#     clf = LogisticRegression(random_state=42, max_iter=300).fit(X[train_mask], y[train_mask])
#     test_score = clf.score(X[test_mask], y[test_mask]) # accuracy
#     train_score = clf.score(X[train_mask], y[train_mask]) # accuracy
#     # print(train_score)
#     return test_score

@torch.no_grad()
def get_cur_train_loss(model):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(args.device)

        emb = model(batch.x, batch.edge_index)
        h_src = emb[batch.edge_label_index[0]]
        h_dst = emb[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        total_loss += float(loss) * pred.size(0)
    return total_loss / data.num_nodes


@torch.no_grad()
def get_cur_test_accuracy(model):
    model.eval()

    emb = model(data.x.to(args.device), data.edge_index.to(args.device))
    X = emb.cpu().detach().numpy()

    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    labels = data.y.cpu().numpy()

    clf = LogisticRegression(random_state=42, max_iter=300).fit(X[train_mask], labels[train_mask])
    test_accuracy = clf.score(X[test_mask], labels[test_mask])

    return test_accuracy

def correlations(y, xs):
    """
    y와 xs의 각 독립 변수에 대해 다양한 상관계수를 계산 (Pearson, Spearman, Kendall, Distance Correlation, Mutual Information, MIC).

    Parameters:
        y (list or numpy array): 종속 변수.
        xs (dict): 독립 변수들의 딕셔너리 (key: 변수 이름, value: 변수 값).

    Returns:
        dict: 각 독립 변수에 대한 모든 상관계수를 포함한 dict.
    """
    def calculate_all_correlations(y, x):
        # Pearson, Spearman, Kendall
        pearson_corr = np.corrcoef(y, x)[0, 1]
        spearman_corr, _ = spearmanr(y, x)
        kendall_corr, _ = kendalltau(y, x)
        
        # Distance Correlation
        distance_corr = distance_correlation(np.array(y), np.array(x))
        
        # Mutual Information
        mutual_info = mutual_info_regression(np.array(x).reshape(-1, 1), np.array(y), random_state=42)
        mutual_info = mutual_info[0]  # mutual_info_regression returns an array
        
        # Maximal Information Coefficient (MIC)
        # mine = MINE()
        # mine.compute_score(x, y)
        # mic = mine.mic()
        
        return pearson_corr, spearman_corr, kendall_corr, distance_corr, mutual_info

    results = {}
    
    for name, x in xs.items():
        pearson, spearman, kendall, distance_corr, mutual_info = calculate_all_correlations(y, x)
        results[name] = {
            "pearson": pearson,
            "spearman": spearman,
            "kendall": kendall,
            "distance_corr": distance_corr,
            "mutual_info": mutual_info,
            # "mic": mic
        }

    return results

def print_correlations(epoch, test_accuracy, loss, explain_rocauc, node_score_var, test_accuracies, correlation_results):
    """
    다양한 상관계수를 출력 (Pearson, Spearman, Kendall, Distance Correlation, Mutual Information, MIC).

    Parameters:
        epoch (int): 현재 에포크.
        test_accuracy (float): 테스트 정확도.
        loss (float): 손실 값.
        explain_rocauc (float): ROC AUC 값.
        node_score_var (float): 노드 분산 값.
        test_accuracies (list): 테스트 정확도 리스트.
        correlation_results (dict): 상관계수 결과 딕셔너리.
    """
    print('---------------')
    print(f'Len {len(test_accuracies)}')
    print(f'Epoch {epoch}, Test Accuracy: {test_accuracy:.3f}, Loss: {loss:.3f}, ROC AUC: {explain_rocauc:.3f}, Node Var: {node_score_var:.3f}')
    print('Correlation Results (Test Accuracy vs ...):')
    print(f"{'Metric':<15} {'Pearson':>10} {'Spearman':>10} {'Kendall':>10} {'Distance':>10} {'Mutual':>10}")
    print('-' * 60)
    for name, values in correlation_results.items():
        print(f"{name:<15} {values['pearson']:>10.3f} {values['spearman']:>10.3f} {values['kendall']:>10.3f} "
              f"{values['distance_corr']:>10.3f} {values['mutual_info']:>10.3f}")
    print('---------------')


def unsup_train_validtest():
    """
    unsupervised setting에서 training 할 때 validation으로
    xaifg explaining score를 쓰는게 나은지 loss를? 쓰는게 나은지
    그니까 뭐가 실제 accuracy랑 더 correlation?이 높은지?

    --- Node ---
    그냥 다 pearson 쓰는게 맞는거같은데
    syn3 syn4 둘 다 나름 잘나옴
    
    - 몇번째까지 해야될지 모르겠음 진동할때의 corr은 좀 명확하지가 않은거같아섣
    - pos가 전체적으로 ㄱㅊ? 이것도 node label쓰거나 하진 않음
    - node기준으로하면 애초에 rocauc빼고 다른 값들은 pos랑 neg랑 다 똑같음..
    
    --- Edge ---
    왜인지는 모르겠는데 edge는 잘안됨..
    """

    saved_epochs = [i for i in range(900,100000)]
    model_weight_path = args.save_path +'_valid_test/' + args.dataset + '_unsupervised_torch_testing'
    
    """
    test_accuracies(node label 사용한 classification,,)와
    losses, explain_rocaucs와의 correlation을 구해야됨
    explain_rocaucs과의 correlation이 더 높게 나오는게 이상적
    """
    test_accuracies = []
    losses = []
    explain_rocaucs = []
    node_score_vars = []
    node_score_kurtosiss = []
    node_score_skews = []
    node_score_entrophies = []
    node_score_bcs = []

    # for line graph plottt
    epoch_score_test_accs = []
    epoch_score_vars = []
    epoch_score_skews = []
    epoch_score_kurtosiss = []  
    epoch_score_train_losses = []
    epoch_score_entropies = []
    epoch_score_bcs = []

    for epoch in saved_epochs:
        if epoch % 180 != 0:
            continue
        matching_files = glob.glob(model_weight_path + f'_epoch{epoch}_*')
        
        if not matching_files:
            continue

        cur_model_weight_path = matching_files[0]
        loss = float(cur_model_weight_path.split('_')[-1])
        
        args.model_weight_path = cur_model_weight_path
        explainer = XAIFG_Node(args=args)

        # explainer.model.eval()
        # emb = explainer.model(data.x.to(args.device), data.edge_index.to(args.device))
        
        # scoring
        test_accuracy = get_cur_test_accuracy(explainer.model)
        # loss = get_cur_train_loss(explainer.model) # 이거 이제 이름에서..

        explain_rocauc, acc, precision, recall = explainer.edge_influences()

        if np.isnan(explainer.node_scores).any():
            continue
        node_score_var = np.mean(explainer.node_scores_vars) # 이렇게하면 각 subgraph에서 node scores의 var의 평균
        # node_score_var = np.var(explainer.node_scores)  # 이렇게하면 모든 subgraph들의 node scores의 var
        node_score_kurtosis = kurtosis(explainer.node_scores, fisher=True)
        node_score_skew = skew(explainer.node_scores)
        hist, bin_edges = np.histogram(explainer.node_scores, bins=10, density=True)
        ent = entropy(hist)
        bc = bimodality_coefficient(explainer.node_scores, node_score_skew, node_score_kurtosis)
        node_score_entrophy = ent

        if np.isnan(explain_rocauc): # rocauc에서 가끔 nan나옴
            continue

        test_accuracies.append(test_accuracy)
        losses.append(loss)
        explain_rocaucs.append(explain_rocauc)
        node_score_vars.append(node_score_var)
        node_score_kurtosiss.append(node_score_kurtosis)
        node_score_skews.append(node_score_skew)
        node_score_entrophies.append(node_score_entrophy)
        node_score_bcs.append(bc)
        
        epoch_score_test_accs.append((epoch, test_accuracy))
        epoch_score_vars.append((epoch, node_score_var))
        epoch_score_skews.append((epoch, node_score_skew))
        epoch_score_kurtosiss.append((epoch, node_score_kurtosis))
        epoch_score_train_losses.append((epoch, loss))
        epoch_score_entropies.append((epoch, node_score_entrophy))
        epoch_score_bcs.append((epoch, bc))
        
        if len(test_accuracies) < 4:
            # 얘네만 출력하고 ㅌㅌ
            print('---------------')
            print(f'Len {len(test_accuracies)}')
            print(f'Epoch {epoch}, Test Accuracy: {test_accuracy:.3f}, Loss: {loss:.3f}')
            print('---------------')
            continue

        if len(test_accuracies) >= 1:
            plot_multiple_line_graphs(
                [epoch_score_test_accs, epoch_score_vars, epoch_score_skews, epoch_score_kurtosiss, epoch_score_entropies, epoch_score_bcs, epoch_score_train_losses],
                ['test accuracy', 'variance', 'skew', 'kurtosis', 'entropy', 'bimodality coefficient', 'train loss'],
                save_path = f'figures/unsup_graphs',
                xtick_interval=4,
                window_size=1 # 2*ws+1개의 평균
            )
            # plot_line_graph(epoch_score_test_accs, 'test accuracy')
            # plot_line_graph(epoch_score_skews, 'skew')
            # plot_line_graph(epoch_score_train_losses, 'train loss')

        print('-----')
        print(epoch_score_test_accs)
        print('-----')
        print(epoch_score_skews)
        print('-----')
        print(epoch_score_bcs)
        print('-----')
        print(epoch_score_train_losses)
        print('-----')
        print(epoch_score_entropies)
        print('-----')

        xs = {
            "Loss": losses,
            "ROC AUC": explain_rocaucs,
            "Node Var": node_score_vars,
            "Node Kurtosis": node_score_kurtosiss,
            "Node Skew": node_score_skews,
            "Node Entrophy": node_score_entrophies,
            "bimodality coefficient": node_score_bcs,
        }


        correlation_results = correlations(test_accuracies, xs)
        print_correlations(epoch, test_accuracy, loss, explain_rocauc, node_score_var, test_accuracies, correlation_results)
        # corr_acc_loss = np.corrcoef(test_accuracies, losses)[0,1]
        # corr_acc_rocauc = np.corrcoef(test_accuracies, explain_rocaucs)[0,1]
        # corr_acc_var = np.corrcoef(test_accuracies, node_score_vars)[0,1]

def normalize_scores(target_list, reference_list):
    # Extract reference scores and determine the reference range.
    _, ref_scores = zip(*reference_list)
    ref_min, ref_max = min(ref_scores), max(ref_scores)
# Extract target scores and determine the target range.
    _, target_scores = zip(*target_list)
    target_min, target_max = min(target_scores), max(target_scores)
# Scale each target score so that target_min maps to ref_min and target_max maps to ref_max.
    normalized_list = [
        (epoch, ref_min + (score - target_min) * (ref_max - ref_min) / (target_max - target_min))
        if target_max > target_min else (epoch, ref_min)
        for epoch, score in target_list
    ]
    return normalized_list

def vis_overlap():
    """
    그냥 일회용임
    """
    epoch_score_test_accs = [(900, 0.6693548387096774), (1080, 0.6693548387096774), (1260, 0.6774193548387096), 
                         (1440, 0.8145161290322581), (1620, 0.8306451612903226), (1800, 0.8870967741935484), 
                         (1980, 0.8951612903225806), (2160, 0.8870967741935484), (2340, 0.8951612903225806), 
                         (2520, 0.8951612903225806), (2700, 0.8951612903225806), (2880, 0.8951612903225806), 
                         (3060, 0.8951612903225806), (3240, 0.8951612903225806), (3420, 0.8951612903225806), 
                         (3600, 0.8951612903225806), (3780, 0.8951612903225806), (3960, 0.8870967741935484), 
                         (4140, 0.8387096774193549), (4320, 0.75)]

    epoch_score_bcs = [(900, 0.8069577625909676), (1080, 0.8100752993065263), (1260, 0.820901859561346), 
                    (1440, 0.8006178998122775), (1620, 0.7494573187596523), (1800, 0.7574971702181915), 
                    (1980, 0.7430324783877577), (2160, 0.7771287105525624), (2340, 0.7715633404478524), 
                    (2520, 0.6986413245472575), (2700, 0.6881926694906773), (2880, 0.678837927085403), 
                    (3060, 0.6834881147629085), (3240, 0.6919281881542559), (3420, 0.6931762420826121), 
                    (3600, 0.7487944994117861), (3780, 0.7785343455674576), (3960, 0.8088294874824318), 
                    (4140, 0.8007181190849267), (4320, 0.7823493020967062)]

    epoch_score_train_losses = [(900, 3.43492), (1080, 3.42276), (1260, 3.40343), (1440, 3.38154), (1620, 3.37986), 
                                (1800, 2.75451), (1980, 2.72395), (2160, 2.28383), (2340, 1.85039), (2520, 1.46808), 
                                (2700, 1.16223), (2880, 1.14708), (3060, 0.902534), (3240, 0.694659), (3420, 0.53944), 
                                (3600, 0.413191), (3780, 0.309119), (3960, 0.237979), (4140, 0.180341), (4320, 0.182607)]

    # epoch_score_entropies = []

    scaled_epoch_score_bcs = normalize_scores(epoch_score_bcs, epoch_score_test_accs)
    scaled_epoch_score_train_losses = normalize_scores(epoch_score_train_losses, epoch_score_test_accs)
    
    # plot_multiple_line_graphs_overlapped(
    #     [epoch_score_test_accs, scaled_epoch_score_train_losses],
    #     ['Test Accuracy', 'Train Loss'],
        
    #     # save_path = f'figures/unsup_graphs',
    #     xtick_interval=4,
    #     window_size=1 # 2*ws+1개의 평균)
    # )
    
    plot_multiple_line_graphs_overlapped(
        [epoch_score_test_accs, scaled_epoch_score_bcs],
        ['Test Accuracy', 'Bimodality Coefficient'],
        
        # save_path = f'figures/unsup_graphs',
        xtick_interval=4,
        window_size=1 # 2*ws+1개의 평균)
    )

if __name__=="__main__":
    """"""
    vis_overlap()
    exit()

    ###### edit here ######
    args.device = 'cuda'
    # xaifg는 cpu가 더 빠를때도 많음

    # node # 여기서 말고 맨 위에서 해야됨
    # args.dataset = 'syn4'
    
    # graph
    # args.dataset = 'BA-2motif'
    # args.hiddens = '25-25-25'
    # args.concat = False
    # args.bn = False

    args.gnn_type = 'unsupervised' # 'supervised' or 'unsupervised'
    # args.hidden_dim = 128 # unsupervised일때만
    args.task = 'pos' # 'pos' or 'neg'
    args.setting = 1 # neg일때 setting 잘보고하기

    
    args.iter = 3
    args.scale = 100000

    # syn3 pos1, neg range(0,511,6)
    # syn4 pos1, neg1
    # ba2  pos1, neg range (0,100)
    #######################

    args.gnn_task = 'node' if args.dataset[:3] == 'syn' else 'graph'

    unsup_train_validtest()