# Environment Setup for HINT-G

This document describes the Python environment and required packages to run the experiments and visualizations for the HINT-G framework.

---

## 🐍 Python Version

- Python 3.11

---

## 📦 Required Packages and Versions

Below are the core packages used in this project:

| Package            | Version                |
|--------------------|------------------------|
| torch              | 2.1.2                  |
| torchvision        | 0.16.2                 |
| torchaudio         | 2.1.2                  |
| torch-geometric    | 2.4.0                  |
| torch-scatter      | 2.1.2+pt21cu121        |
| torch-sparse       | 0.6.18+pt21cu121       |
| torch-cluster      | 1.6.3+pt21cu121        |
| torch-spline-conv  | 1.2.2+pt21cu121        |
| scikit-learn       | 1.7.1                  |
| matplotlib         | 3.10.3                 |
| networkx           | 3.5                    |
| tqdm               | 4.67.1                 |

---

## Installing PyG Extensions (Required)

The following packages must be installed from the official PyG wheel index due to CUDA and PyTorch version compatibility:

```bash
pip install torch-scatter==2.1.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

pip install torch-sparse==0.6.18+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

pip install torch-cluster==1.6.3+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

pip install torch-spline-conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

You must install these **after** installing PyTorch 2.1.2 with CUDA 12.1.

---

## Installation Example

#### Create environment (optional)
```bash
conda create -n hint_g_env python=3.11
conda activate hint_g_env
```

#### Install core packages
```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

#### Install PyG extensions
```bash
pip install torch-scatter==2.1.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-sparse==0.6.18+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-cluster==1.6.3+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-spline-conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

#### Install remaining packages
```bash
pip install torch-geometric==2.4.0 matplotlib==3.10.3 scikit-learn==1.7.1 networkx==3.5 tqdm==4.67.1
```
---

## 💡 Notes
- This environment has been tested and used to reproduce all results and visualizations from the paper.