import numpy as np
import matplotlib.pyplot as plt

# 데이터 설정
num_edges = np.array([1, 2, 3, 4, 5, 10])

# topk num
data = {
    "Original": [0.919, 0.919, 0.919, 0.919, 0.919, 0.919],
    "Random": [0.923, 0.915, 0.908, 0.906, 0.898, 0.847],
    # "Random homophilic": [0.920, 0.916, 0.905, 0.905, 0.903, 0.834],
    # "Random heterophilic": [0.920, 0.915, 0.909, 0.907, 0.895, 0.819],
    "XAIFG_Node": [0.883, 0.833, 0.620, 0.602, 0.579, 0.546],
    "XAIFG_Edge": [0.931, 0.852, 0.750, 0.630, 0.606, 0.602]
}

# topk percent
# data = {
#     "Original": [0.919, 0.919, 0.919, 0.919, 0.919, 0.919],
#     "Random": [0.917, 0.909, 0.910, 0.899, 0.882, 0.829],
#     # "Random homophilic": [0.915, 0.912, 0.909, 0.901, 0.892, 0.839],
#     # "Random heterophilic": [0.919, 0.910, 0.903, 0.899, 0.889, 0.826],
#     "XAIFG_Node": [0.852, 0.748, 0.606, 0.591, 0.579, 0.480],
#     "XAIFG_Edge": [0.852, 0.832, 0.833, 0.630, 0.604, 0.580]
# }

# 그래프 생성
plt.figure(figsize=(8, 6))

markers = ['o', 's', 'D', '^', 'v', 'x']
for (label, values), marker in zip(data.items(), markers):
    plt.plot(num_edges, values, marker=marker, label=label, markersize=10)

plt.xlabel("Number of Negative Edges", fontsize=18) #
plt.ylabel("Test Accuracy", fontsize=18)
plt.title("Effect of Adding Top-K Negative Edges", fontsize=20) #
plt.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()
