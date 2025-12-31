# nCLG 

nCLG 是一种基于 **速率失真理论 (Rate-Distortion Theory)** 和 **非对比学习范式** 的增强型图协同过滤模型。它通过在标准 LightGCN 之上引入显式的几何空间约束，有效解决了图神经网络在推荐系统中的 **维度坍缩 (Dimensional Collapse)** 和 **过度平滑 (Over-smoothing)** 问题。

---

## 核心设计逻辑

传统的推荐模型（如 LightGCN）依赖 BPR 损失进行排序学习，在训练中后期极易导致 Embedding 空间坍缩至极低维子空间。nCLG 抛弃了负采样 BPR 模式，采用 **Alignment（对齐）+ Compactness（紧凑）** 的双重机制：

1.  **Alignment (对齐拉力)**：通过 L2 归一化将嵌入映射至超球面，强制正样本对（User-Item）相互接近。
2.  **Compactness (紧凑推力)**：
    *   **局部压缩 (Local Compression)**：利用预计算的图拓扑共现邻居（Co-occurrence Neighbors），使具有二阶语义关联的节点在局部聚集成簇。
    *   **全局扩张 (Global Expansion)**：利用 `logdet` 函数最大化全局嵌入占据的空间体积，防止特征趋同，提高模型分辨率。

---

## 主要特性

- **非对比学习范式**：无需显式的负采样，通过全局速率（Rate）约束实现隐式排斥，训练更稳定。
- **注意力加权邻域**：支持 Attentive Compactness，动态识别并抑制共现图中的流行度噪声（Popularity Bias）。
- **非对称扩张优化**：针对物品端（Item）更强的坍缩引力，提供独立的排斥力增强方案。

---


## 目录结构

- **src/models/general/nCLG.py**：nCLG 模型实现。
- **./data/**：实验数据
- **./logs/**：日志文件


---

## 环境要求

- pip install -r requirements.txt
- Python 3.10+
- PyTorch 2.0+
- SciPy, NumPy, tqdm, matplotlib
- [ReChorus](https://github.com/THUwangcy/ReChorus) 实验框架

---

## 参数指南

| 参数 | 默认值 | 说明 | 调参建议 |
| :--- | :--- | :--- | :--- |
| `alpha` | 0.05 | 紧凑性损失的总权重 | **核心参数**。过大会导致白噪化（谱线全平），过小会导致坍缩。 |
| `epsilon` | 1.0 | 失真常数（介电常数） | 控制扩张梯度的“锐度”。减小会增加全局排斥力。 |
| `temp` | 0.2 | 温度系数 | 影响对齐精度和注意力分布。 |
| `neighbor_topk`| 20 | 共现邻居数量 | 捕捉二阶语义边界。 |
| `item_multi` | 1.0 | 物品端扩张加权倍数 | 用于平衡 Item 侧严重的流行度黑洞效应。 |
| `num_seeds` | 20 | 每批次采样种子数 | 增加该值可稳定局部约束的梯度，但会略微增加计算量。 |

---

## 奇异值曲线（可选）

.\main.py  
line 105-135

---


### 训练命令示例
```bash
python main.py --model_name nCLG \
                --emb_size 64 \
                ----batch_size 4096 \ 
                --n_layers 3 \
                --lr 5e-4 \
                --l2 1e-8 \
                --dataset "Grocery_and_Gourmet_Food" \
                --temp 0.2 \
                --alpha 0.039 \ 
                --epsilon 1.1 
