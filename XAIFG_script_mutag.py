import torch

from models import XAIFG

from utils import args

"""
iteration   20
scale       ?
damp        ?
"""

if __name__== '__main__':

    ###### edit here ######
    args.device = 'cuda'

    args.dataset = 'Mutagenicity'

    args.gnn_type = 'supervised' # 'supervised' or 'unsupervised'
    args.task = 'pos' # 'pos' or 'neg'
    args.bn = False
    args.concat = False
    #######################

    args.gnn_task = 'node' if args.dataset[:3] == 'syn' else 'graph'
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    args.model_weight_path = args.save_path + args.dataset +'_'+ args.gnn_type

    explainer = XAIFG(args=args)

    rocauc = explainer.edge_influences()
    print(rocauc)

    #### Accuracy
    # data = explainer.data
    # nodes = data.nodes
    # correct = 0
    # cnt = 0
    # from tqdm import tqdm
    # for nodeid in tqdm(nodes):
    #     adj, fea, label = data.adjs[nodeid], data.feas[nodeid], data.labels[nodeid]
    #     label = torch.tensor(label)
    #     fea_tensor = torch.tensor(fea, dtype=torch.float32).unsqueeze(0)
    #     label_tensor = torch.argmax(label).unsqueeze(0).to(args.device)

    #     output = explainer.model((fea_tensor, torch.tensor(adj, dtype=torch.float32).unsqueeze(0)))
        
    #     pred = torch.argmax(output, dim=-1).item()
    #     lb = torch.argmax(label, dim=-1)
    #     if pred == lb:
    #         correct += 1
    #     cnt += 1
    # print(f'accuracy: {correct/cnt}')
