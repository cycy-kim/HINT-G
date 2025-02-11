"""
GNN 모델 training하는 script
explainer 학습하는거 아님
"""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import torch_geometric
import torch_geometric.utils as torch_geometric_utils

from sklearn.linear_model import LogisticRegression
import numpy as np

import pickle as pkl

from models import GCN_Emb
from dataset import Data, LinkNeighborLoader
from utils import args


'''TODO
그냥 epoch 지날수록 negative sample ratio를 줄이자...
'''

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)

args.epochs = 100000 #500
args.lr = 0.001 # lr 0.01로 하면 ㄱㅊ긴한데 그래프가 잘안보임
args.weight_decay= 5e-4 #5e-4
# args.weight_decay= 0.0 #5e-4 # for overfitting

args.dataset='syn4'
args.hidden_dim = 128 # 원래는 128 # 마지막 node emb dim을 더 줄여보기 -> logistic regression이 너무 강한상황일수도??
args.num_layers = 3

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
# data.train_mask = val_mask # val을 train으로..
# data.val_mask = test_mask
# data.test_mask = train_mask # test는 빡세게


# train_data = data.subgraph(data.train_mask) # 일부만 train data로..

torch_geometric.seed_everything(0)
cur_neg_sampling_ratio = 1.0
train_loader = LinkNeighborLoader(data, batch_size=99999, shuffle=True,
                                neg_sampling_ratio=cur_neg_sampling_ratio, num_neighbors=[10,10]) # 1e-6이면 거의 안하지않나
# train_loader = LinkNeighborLoader(data, batch_size=128, shuffle=True,
#                                 neg_sampling_ratio=1.0,num_neighbors=[10,10])


### 의도적으로 overfitting 연출하려고
model = GCN_Emb(data.x.shape[1],
                hidden_channels = args.hidden_dim,
                num_layers = args.num_layers,
                # out_channels=16,
                # act = None,
                ).to(args.device)

### 멀쩡버전
# model = GCN_Emb(data.x.shape[1], hidden_channels=args.hidden_dim, num_layers=3).to(args.device)


def train_one_epoch(model, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(args.device)
        optimizer.zero_grad()

        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)
    return total_loss / data.num_nodes



# @torch.no_grad()
def test(out):
    X = out.cpu().detach().numpy() # (num_nodes, d)
    # X = out
    # node classification

    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    labels = data.y.cpu().numpy()
    
    clf = LogisticRegression(random_state=42, max_iter=300).fit(X[train_mask], labels[train_mask])
    test_accuracy = clf.score(X[test_mask], labels[test_mask])
    # print(labels[test_mask].shape)
    # print(np.sum(labels[test_mask]))
    # # 이게 0.588
    # exit()
    return test_accuracy

def train_embedder():
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, step_size=300, gamma=0.5)
    
    best_train_loss = float('inf')
    best_score = float('-inf')
    before_score = 0
    patience = 0
    eps = 1e-4
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer)
        # scheduler.step()
        
        with torch.no_grad():
            model.eval()
            emb = model(data.x.to(args.device), data.edge_index.to(args.device))
        
        # if train_loss <= best_train_loss + eps:
        # if train_loss <= best_train_loss + eps:
        if epoch % 20 == 0:
            if train_loss < best_train_loss:
                best_train_loss=train_loss
            score = test(emb)
            print(f'{args.dataset}, {epoch:4}, {train_loss:6.4}, {score:6.4}')

            cur_model_weight_path = model_weight_path + f'_epoch{epoch}_{train_loss:.6}'
            torch.save(model.state_dict(), cur_model_weight_path)

            if score <= 0.8:
                continue
            
            # if score > 0.9:
            if best_score >= score:
                patience += 1
            else:
                best_score = score
                patience = 0
            global cur_neg_sampling_ratio
            if patience >= 10 and cur_neg_sampling_ratio != 1e-6 and score >= 0.7:
                global train_loader
                cur_neg_sampling_ratio *= 0.7
                if cur_neg_sampling_ratio <= 0.5 and args.lr != 0.003:
                    args.lr = 0.003
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr , weight_decay=args.weight_decay)
                # if score >= 0.9 and cur_neg_sampling_ratio >= 0.1:
                #     cur_neg_sampling_ratio *= 0.5
                    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr , weight_decay=args.weight_decay)
        

                if cur_neg_sampling_ratio <= 0:
                    cur_neg_sampling_ratio = 1e-6
                print(f'neg sampling ratio decreased to {cur_neg_sampling_ratio}')
                train_loader = LinkNeighborLoader(data, batch_size=99999, shuffle=True,
                                                neg_sampling_ratio=cur_neg_sampling_ratio, num_neighbors=[10,10]) # 1e-6이면 거의 안하지않나
                # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                patience = 0
            
            # before_score = score
        elif epoch % 200 == 0:
            """"""
            # if cur_neg_sampling_ratio == 1e-6:
            #     score = test(emb)
            #     print(f'{args.dataset}, {epoch:4}, {train_loss:6.4}, {score:6.4}')
            #     cur_model_weight_path = model_weight_path + f'_epoch{epoch}_{train_loss:.6}'
            #     torch.save(model.state_dict(), cur_model_weight_path)
            # else:
            # print(f'-----{args.dataset}, {epoch:4}, {train_loss:6.4}, {score:6.4}')


        # if epoch % 20 ==0 and score > 0.8:
        # if score >= 0.7:
        # if epoch % 60 == 0:
        #     cur_model_weight_path = model_weight_path + f'_epoch{epoch}'
        #     torch.save(model.state_dict(), cur_model_weight_path)

        # if score > best_score:
        #     best_score = score
        #     torch.save(model.state_dict(), model_weight_path)
        #     # print(f'Best model saved at Epoch: {epoch:03d} with Loss: {loss:.4f}')
        #     print(f'Best model saved at Epoch: {epoch:03d} with Score: {score:.4f}')

        #     # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Best Loss: {best_loss:.4f}')

    model.eval()
    emb = model(data.x.to(args.device), data.edge_index.to(args.device))
    return emb

if __name__=="__main__":
    emb = train_embedder()
    # emb = np.load('./syn3_graphsage_node_emb.npy')
    score = test(emb)
    print(score)
    # torch.save(emb, './emb_syn3_uns3.pt')
    # train_classifier()
    # test_classifier_subgraph()
    