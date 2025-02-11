import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_pos_adj_node(adj, node_scores, show=True, save=False, pic_name=''):
    """Node score기준으로 노드크기만 다르게 edge들은 다 똑같이.. 

    Args:
        adj (_type_): _description_
        node_scores (_type_): _description_
        show (bool, optional): _description_. Defaults to True.
        save (bool, optional): _description_. Defaults to False.
        pic_name (str, optional): _description_. Defaults to ''.
    """
    adj = np.array(adj)
    G = nx.from_numpy_array(adj)

    # 고립된 노드 제거
    G.remove_nodes_from(list(nx.isolates(G)))

    plt.figure(figsize=(9, 9))

    # 노드 크기 조정 (로그 스케일 변환)
    min_size, max_size = 300, 1500  # 노드 크기 범위
    node_scores = np.array(node_scores, dtype=np.float64)

    min_score, max_score = node_scores.min(), node_scores.max()

    if max_score > min_score:
        # scaled_scores = np.log(0.1 + node_scores - min_score)  # log로 스케일링
        scaled_scores = node_scores         # 그대로 linear하게 스케일링

        # 0~1 to linearly align to node size
        normalized_scores = (scaled_scores - scaled_scores.min()) / (scaled_scores.max() - scaled_scores.min())
        node_sizes = normalized_scores * (max_size - min_size) + min_size

    else:
        node_sizes = np.full_like(node_scores, (max_size + min_size) / 2)  # 모든 노드 크기 동일

    # 노드 위치 결정
    pos = nx.spring_layout(G)

    # 그래프 시각화 (엣지는 기본 설정, 노드 크기만 다르게)
    nx.draw(G, pos, with_labels=False, node_color='skyblue', edge_color='black', width=1, 
            node_size=node_sizes, font_size=12, font_color='darkred', edgecolors='darkblue', linewidths=1.5)

    plt.title('Graph Visualization from Adjacency Matrix (Log Scale)')

    if save:
        plt.savefig(f'pics/graph_{pic_name}.png') 
    if show:
        plt.show()

    plt.close()

def plot_pos_adj_both(adj, node_scores, special_edges=[], topk=0, 
                        show=True, save=False, pic_name='',
                        node_min_size=300, node_max_size=1500, 
                        edge_weight_multiplier=9, edge_weight_offset=2,
                        with_labels=True):
    """
    노드 크기는 node_scores에 따라, 엣지 두께는 special_edges의 score에 따라 조절하여 그래프를 시각화합니다.
    special_edges가 제공되면, score에 따라 min-max normalization 후 topk 개의 엣지를 강조(빨간색)하고,
    나머지 special edge는 검정색으로, 그 외 엣지는 기본 두께(1.0)로 표시합니다.
    
    Args:
        adj (array-like): 인접행렬.
        node_scores (list or array): 각 노드의 점수 (노드 크기 조절용).
        special_edges (list of tuple, optional): [(score, (src, des)), ...] 형식의 리스트.
        topk (int, optional): 강조할 상위 special edge 개수 (0이면 강조 없음).
        show (bool, optional): 그래프를 화면에 출력할지 여부.
        save (bool, optional): 그래프를 이미지 파일로 저장할지 여부.
        pic_name (str, optional): 저장할 파일명 접미사.
        node_min_size (int, optional): 최소 노드 크기.
        node_max_size (int, optional): 최대 노드 크기.
        edge_weight_multiplier (float, optional): 엣지 두께 계산 시 곱할 상수.
        edge_weight_offset (float, optional): 엣지 두께 계산 시 더할 상수.
        with_labels (bool, optional): 노드 라벨 표시 여부.
    """
    # 1. 그래프 생성 및 고립 노드 제거
    adj = np.array(adj)
    G = nx.from_numpy_array(adj)
    
    # 인접행렬에 있는 노드 번호와 일치하도록, 고립 노드를 제거한 후의 노드 리스트 (정렬)
    nodes_in_G = sorted(G.nodes())
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # 2. Figure 생성
    plt.figure(figsize=(9, 9))
    
    # 3. 노드 크기 조절 (node_scores 기반)
    # node_scores는 인접행렬의 인덱스 순서와 맞다고 가정합니다.
    node_scores_arr = np.array(node_scores, dtype=np.float64)
    # 고립 노드를 제거한 후 남은 노드에 해당하는 score만 선택
    nodes_in_G = sorted(G.nodes())
    if len(nodes_in_G) > 0:
        node_scores_filtered = node_scores_arr[nodes_in_G]
    else:
        node_scores_filtered = node_scores_arr

    if node_scores_filtered.size > 0:
        min_score_val = node_scores_filtered.min()
        max_score_val = node_scores_filtered.max()
    else:
        min_score_val, max_score_val = 0, 1

    if max_score_val > min_score_val:
        normalized_scores = (node_scores_filtered - min_score_val) / (max_score_val - min_score_val)
        node_sizes = normalized_scores * (node_max_size - node_min_size) + node_min_size
    else:
        node_sizes = np.full_like(node_scores_filtered, (node_max_size + node_min_size) / 2)
    
    # 4. 엣지 두께 조절 (special_edges 기반)
    if special_edges:
        # special_edges의 score들을 min-max 정규화
        scores = [score for score, _ in special_edges]
        min_edge_score, max_edge_score = min(scores), max(scores)
        if max_edge_score > min_edge_score:
            normalized_edges = [((score - min_edge_score) / (max_edge_score - min_edge_score), edge) 
                                for score, edge in special_edges]
        else:
            normalized_edges = [(1.0, edge) for score, edge in special_edges]
    else:
        normalized_edges = []
    
    # topk 설정: topk>0이면 상위 topk special edge를 별도 처리
    if topk > 0:
        # top_special_edges = sorted(normalized_edges, key=lambda x: x[0], reverse=True)[:topk]
        top_special_edges = normalized_edges[:topk]
    else:
        top_special_edges = []
    top_special_edges_set = set((src, des) for _, (src, des) in top_special_edges)
    
    # 엣지별 두께와 색상 결정
    edge_colors = []
    edge_weights = []
    for u, v in G.edges():
        if (u, v) in top_special_edges_set or (v, u) in top_special_edges_set:
            # 상위 topk special edge: 빨간색, score에 따라 두께 결정
            score = next(score for score, (src, des) in top_special_edges 
                         if (src == u and des == v) or (src == v and des == u))
            edge_weight = score * edge_weight_multiplier + edge_weight_offset
            edge_colors.append('red')
        elif any((u == src and v == des) or (u == des and v == src) for _, (src, des) in normalized_edges):
            # 일반 special edge: 검정색, score에 따라 두께 결정
            score = next(score for score, (src, des) in normalized_edges 
                         if (src == u and des == v) or (src == v and des == u))
            edge_weight = score * edge_weight_multiplier + edge_weight_offset
            edge_colors.append('black')
        else:
            # 일반 엣지: 기본 두께와 색상
            edge_weight = 1.0
            edge_colors.append('black')
        edge_weights.append(edge_weight)
    
    # 5. 레이아웃 결정
    pos = nx.spring_layout(G)
    
    # 6. 그래프 그리기  
    # 노드 리스트와 노드 크기는 동일 순서를 맞춰줍니다.
    nx.draw(G, pos,
            nodelist=nodes_in_G,
            with_labels=with_labels,
            node_color='skyblue',
            edge_color=edge_colors,
            width=edge_weights,
            node_size=list(node_sizes),  # node_sizes는 nodes_in_G 순서에 맞게 계산됨.
            font_size=12,
            font_color='darkred',
            edgecolors='darkblue',
            linewidths=1.5)
    
    plt.title('Combined Graph Visualization')
    
    # 7. 저장 또는 화면 출력
    if save:
        save_path = 'pics'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_save_path = os.path.join(save_path, f'graph_{pic_name}.png')
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {full_save_path}")
    if show:
        plt.show()
    
    plt.close()

def plot_neg_adj_both(adj, node_scores, special_edges=[], topk=0, show=True, save=False, graph_index=None, pic_name='',
                 node_min_size=300, node_max_size=1500):
    """
    인접행렬(adj)과 node_scores를 기반으로 기본 그래프(adj의 edge)를 그린 후,
    special_edges (이미 정렬되어 있음, adj에는 없는 edge)를 dashed line으로 추가로 표시합니다.
    상위 topk special edge는 빨간색 dashed line으로 강조합니다.
    
    Args:
        adj (Tensor or array-like): 인접행렬 (Tensor인 경우 .cpu()로 변환)
        node_scores (list or array): 각 노드의 점수 (노드 크기 조절용)
        special_edges (list of tuple, optional): [(score, (src, des)), ...] 형태의 리스트.
                                                  이들은 adj에 없는 추가 edge임.
        topk (int, optional): special_edges 중 상위 몇 개를 빨간색 dashed line으로 표시할지 (0이면 없음)
        show (bool, optional): 그래프를 화면에 출력할지 여부.
        save (bool, optional): 그래프를 이미지 파일로 저장할지 여부.
        graph_index: (사용하지 않음; 호환성 용)
        pic_name (str, optional): 저장 시 파일명 접미사.
        node_min_size (int, optional): 노드 크기의 최소값.
        node_max_size (int, optional): 노드 크기의 최대값.
    """
    # 만약 adj가 tensor라면 numpy array로 변환
    if hasattr(adj, 'cpu'):
        adj = np.array(adj.cpu())
    else:
        adj = np.array(adj)
    
    # 인접행렬로부터 그래프 생성 (adj의 edge들은 그대로 사용)
    G = nx.from_numpy_array(adj)
    # 고립된 노드 제거
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # 노드 크기를 node_scores에 따라 계산 (min-max 정규화)
    node_scores_arr = np.array(node_scores, dtype=np.float64)
    nodes = list(G.nodes())
    # 인접행렬의 인덱스와 노드 번호가 일치한다고 가정
    node_scores_filtered = node_scores_arr[nodes]
    if node_scores_filtered.size > 0:
        min_score = node_scores_filtered.min()
        max_score = node_scores_filtered.max()
        if max_score > min_score:
            normalized = (node_scores_filtered - min_score) / (max_score - min_score)
            node_sizes = normalized * (node_max_size - node_min_size) + node_min_size
        else:
            node_sizes = np.full_like(node_scores_filtered, (node_max_size + node_min_size) / 2)
    else:
        node_sizes = np.full(len(nodes), node_min_size)
    
    # 예시로 노드 색상을 정함 (원하는 방식으로 수정 가능)
    # Figure 생성 및 노드 위치 결정 (모든 그리기에 동일한 위치 사용)
    plt.figure(figsize=(9, 9))
    pos = nx.spring_layout(G)
    
    # 먼저, 기존 adj의 edge들을 그대로 그림
    nx.draw(G, pos, with_labels=False,
            node_color='skyblue',
            edge_color='black',
            width=1.0,
            node_size=node_sizes,
            font_size=15,
            font_color='darkred',
            edgecolors='darkblue',
            linewidths=1.5)
    
    # special_edges 처리  
    # special_edges는 이미 정렬되어 있다고 가정하고, 상위 topk edge만 빨간색으로 표시
    top_special_edges = special_edges[:topk]
    
    top_special_edges_set = set((min(src, des), max(src, des)) for _, (src, des) in top_special_edges)
    
    special_edges_to_draw = [(src, des) for src, des in top_special_edges_set]
    # 그래프에 임시로 추가 (시각화를 위해)
    G.add_edges_from(special_edges_to_draw)
    
    # 각 special edge를 dashed line으로 그림 (topk에 해당하면 빨간색, 그 외는 검정색)
    for src, des in top_special_edges_set:
        color = 'red'
        nx.draw_networkx_edges(G, pos, edgelist=[(src, des)], edge_color=color, style='dashed', width=4)
    
    plt.title('Graph Visualization from Adjacency Matrix')
    
    if save:
        save_path = 'pics'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'graph_{pic_name}.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    
    plt.close()

def plot_line_graph(scores, name=''):
    epochs, values = zip(*scores)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker='o', linestyle='-', color='b', label=name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.show()

# def plot_multiple_line_graphs(scores_list, names=None, save_path=None):
#     """
#     여러 개의 (epoch, score) 데이터를 받아 세로로 쭉 나열하여 개별 그래프를 생성.
#     저장 경로가 제공되면 해당 경로에 저장.

#     Args:
#         scores_list (List of List of Tuple): [[(epoch, score), ...], [(epoch, score), ...], ...]
#         names (List of str, optional): 그래프 제목 리스트 (None일 경우 자동 생성)
#         save_path (str, optional): 그래프를 저장할 경로 (None이면 저장하지 않음)
#     """
#     num_graphs = len(scores_list)
    
#     plt.figure(figsize=(8, 4 * num_graphs))  # 전체 크기 조정 (세로 길이 늘리기)

#     for i, scores in enumerate(scores_list):
#         epochs, values = zip(*scores)  # (epoch, score) 튜플 분리
#         name = names[i] if names and i < len(names) else f"Graph {i+1}"  # 그래프 이름 설정

#         plt.subplot(num_graphs, 1, i + 1)  # 세로로 배치
#         plt.plot(epochs, values, marker='o', linestyle='-', color='b', label=name)
#         plt.xlabel("Epoch")
#         plt.ylabel("Score")
#         plt.title(name)
#         plt.legend()
#         plt.grid(True)

#     plt.tight_layout()  # 그래프 간 간격 조정

#     # 저장 경로가 있으면 이미지 파일 저장
#     if save_path:
#         max_epoch = max(epoch for epoch, _ in scores_list[0]) # 최대 epoch 찾기
#         save_filename = f"{max_epoch}_graph.png"
#         save_full_path = os.path.join(save_path, save_filename)

#         plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
#         print(f"Graph saved to {save_full_path}")
#     else:
#         plt.show()

#     plt.close()

# def plot_multiple_line_graphs(scores_list, names=None, save_path=None):
#     """
#     여러 개의 (epoch, score) 데이터를 받아 세로로 배치하여 개별 그래프를 생성.
#     X축 간격을 균등하게 조정하여 정렬된 그래프를 표시.
#     저장 경로가 제공되면 해당 경로에 저장.

#     Args:
#         scores_list (List of List of Tuple): [[(epoch, score), ...], [(epoch, score), ...], ...]
#         names (List of str, optional): 그래프 제목 리스트 (None일 경우 자동 생성)
#         save_path (str, optional): 그래프를 저장할 경로 (None이면 저장하지 않음)
#     """
#     num_graphs = len(scores_list)
    
#     plt.figure(figsize=(8, 4 * num_graphs))  # 전체 크기 조정 (세로 길이 늘리기)

#     for i, scores in enumerate(scores_list):
#         epochs, values = zip(*scores)  # (epoch, score) 튜플 분리
#         name = names[i] if names and i < len(names) else f"Graph {i+1}"  # 그래프 이름 설정

#         x_indices = list(range(len(epochs)))  # X축을 균등한 간격으로 정렬된 정수 인덱스로 변환

#         plt.subplot(num_graphs, 1, i + 1)  # 세로로 배치
#         plt.plot(x_indices, values, marker='o', linestyle='-', color='b', label=name)
#         plt.xticks(x_indices, epochs)  # 원래 epoch 값을 x축 눈금으로 표시
#         plt.xlabel("Epoch")
#         plt.ylabel("Score")
#         plt.title(name)
#         plt.legend()
#         plt.grid(True)

#     plt.tight_layout()  # 그래프 간 간격 조정

#     # 저장 경로가 있으면 이미지 파일 저장
#     if save_path:
#         max_epoch = max(epoch for epoch, _ in scores_list[0]) # 최대 epoch 찾기
#         save_filename = f"{max_epoch}_graph.png"
#         save_full_path = os.path.join(save_path, save_filename)

#         plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
#         print(f"Graph saved to {save_full_path}")
#     else:
#         plt.show()

#     plt.close()

def moving_average(values, k):
    """값 리스트에 대해 앞뒤 k개를 포함한 이동 평균을 계산"""
    smoothed_values = []
    for i in range(len(values)):
        start = max(0, i - k)
        end = min(len(values), i + k + 1)
        smoothed_values.append(np.mean(values[start:end]))
    return smoothed_values

def plot_multiple_line_graphs(scores_list, names=None, save_path=None, xtick_interval=4, window_size=0):
    """
    여러 개의 (epoch, score) 데이터를 받아 세로로 배치하여 개별 그래프를 생성.
    X축 간격을 균등하게 조정하여 정렬된 그래프를 표시.
    저장 경로가 제공되면 해당 경로에 저장.

    Args:
        scores_list (List of List of Tuple): [[(epoch, score), ...], [(epoch, score), ...], ...]
        names (List of str, optional): 그래프 제목 리스트 (None일 경우 자동 생성)
        save_path (str, optional): 그래프를 저장할 경로 (None이면 저장하지 않음)
        window_size (int): 앞뒤로 평균 낼 개수(smoothing 할때), window_size=0이면 smoothing 없다는 뜻 (기본값: 0)
        xtick_interval (int): X축 레이블을 몇 개마다 하나씩 표시할지 (기본값: 4)
    """
    num_graphs = len(scores_list)
    
    plt.figure(figsize=(8, 4 * num_graphs))  # 전체 크기 조정 (세로 길이 늘리기)

    for i, scores in enumerate(scores_list):
        epochs, values = zip(*scores)  # (epoch, score) 튜플 분리
        smoothed_values = moving_average(values, window_size)  # 이동 평균 적용
        
        name = names[i] if names and i < len(names) else f"Graph {i+1}"  # 그래프 이름 설정

        x_indices = list(range(len(epochs)))  # X축을 균등한 간격으로 정렬된 정수 인덱스로 변환

        plt.subplot(num_graphs, 1, i + 1)  # 세로로 배치
        plt.plot(x_indices, smoothed_values, marker='o', linestyle='-', color='b', label=f"{name} (smoothed)")

        # X축 눈금 (epoch)에서 4개 중 1개씩만 표시
        filtered_epochs = [epoch if idx % xtick_interval == 0 else '' for idx, epoch in enumerate(epochs)]
        plt.xticks(x_indices, filtered_epochs)  

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(name)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()  # 그래프 간 간격 조정

    # 저장 경로가 있으면 이미지 파일 저장
    if save_path:
        max_epoch = max(epoch for epoch, _ in scores_list[0])  # 최대 epoch 찾기
        save_filename = f"{max_epoch}_graph.png"
        save_full_path = os.path.join(save_path, save_filename)

        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_full_path}")
    else:
        plt.show()

    plt.close()


def plot_multiple_line_graphs_overlapped(scores_list, names=None, save_path=None, xtick_interval=4, window_size=0):
    """
    여러 개의 (epoch, score) 데이터를 하나의 그래프에 겹쳐서 표시합니다.
    X축 간격을 균등하게 조정하여 정렬된 그래프를 표시하며,
    저장 경로가 제공되면 해당 경로에 저장합니다.
    
    Args:
        scores_list (List of List of Tuple): [[(epoch, score), ...], [(epoch, score), ...], ...]
        names (List of str, optional): 그래프 제목 리스트 (None일 경우 자동 생성)
        save_path (str, optional): 그래프를 저장할 경로 (None이면 저장하지 않음)
        window_size (int): 앞뒤로 평균 낼 개수 (smoothing 할 때), window_size=0이면 smoothing 없이 원본 값 사용 (기본값: 0)
        xtick_interval (int): X축 레이블을 몇 개마다 하나씩 표시할지 (기본값: 4)
    """
    num_graphs = len(scores_list)
    
    # 단일 그래프에 모든 데이터를 겹쳐서 그리므로, 세로 크기는 고정합니다.
    plt.figure(figsize=(10, 4))

    markers = ['o', 's', 'D', '^', 'v', 'x']
    for i, scores in enumerate(scores_list):
        epochs, values = zip(*scores)  # (epoch, score) 튜플 분리
        smoothed_values = moving_average(values, window_size)  # 이동 평균 적용
        name = names[i] if names and i < len(names) else f"Graph {i+1}"  # 그래프 이름 설정
        x_indices = list(range(len(epochs)))  # X축을 균등한 간격으로 정렬된 정수 인덱스로 변환

        plt.plot(x_indices, smoothed_values, marker=markers[i], linestyle='-', label=f"{name}", markersize=10)

    # X축 눈금 설정 (마지막으로 처리된 epochs 사용; 모든 데이터의 epoch 값이 동일하다고 가정)
    filtered_epochs = [epoch if idx % xtick_interval == 0 else '' for idx, epoch in enumerate(epochs)]
    plt.xticks(x_indices, filtered_epochs)

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Test Accuracy", fontsize=18)
    plt.title(f"{names[0]} vs {names[1]}, (Corr. = -0.812)", fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        max_epoch = max(epoch for epoch, _ in scores_list[0])  # 첫 번째 데이터의 최대 epoch 사용
        save_filename = f"{max_epoch}_graph.png"
        save_full_path = os.path.join(save_path, save_filename)
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_full_path}")
    else:
        plt.show()

    plt.close()



# 그냥 개간단버전
def plot_adj(adj, show=True, save=False, graph_index=None):
    adj = np.array(adj)
    G = nx.from_numpy_array(adj)
    plt.figure(figsize=(5, 5))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=300, edge_color='black', font_size=15, font_color='darkred')
    plt.title('Graph Visualization from Adjacency Matrix')
    if save:
        plt.savefig(f'pics/graph_{graph_index}.png') 
    if show:
        plt.show()
    plt.close()
