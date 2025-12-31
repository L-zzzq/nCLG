import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.general.LightGCN import *

# ============================================================================
# 1. IPOT 求解器：用于求解最优传输矩阵 Q* (对应论文 Eq. 9)
# ============================================================================
class IPOTSolver(nn.Module):
    def __init__(self, K, iterations=20, beta=0.5):
        super().__init__()
        self.K = K  # 聚类数
        self.iterations = iterations
        self.beta = beta # 逆温度系数

    @torch.no_grad() # 根据包络定理，IPOT不需要传导梯度
    def forward(self, C):
        """
        C: 成本矩阵 (在这里是 Embedding 与 Centroids 的余弦相似度)
        返回 Q*: 满足边际分布约束的最优分配矩阵
        """
        n = C.shape[0]
        # 初始化传输矩阵为均匀分布
        Q = torch.ones(n, self.K, device=C.device) / (n * self.K)
        
        # 目标分布约束: 行之和为 1/n, 列之和为 1/K
        a = torch.ones(n, 1, device=C.device) / n
        b = torch.ones(self.K, 1, device=C.device) / self.K
        
        # 这里的核矩阵为 exp(C / beta)
        K_mat = torch.exp(C / self.beta)
        
        # IPOT 迭代更新 (不需要像 Sinkhorn 那样计算梯度，速度极快)
        sigma = torch.ones(self.K, 1, device=C.device) / self.K
        for _ in range(self.iterations):
            # q_yu = sigma_y * K_yu
            Q = K_mat * sigma.t()
            # 归一化行边际分布
            Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12) * a
            # 更新列缩放因子 sigma
            sigma = b / (torch.mm(Q.t(), torch.ones(n, 1, device=C.device)) + 1e-12)
            
        # 返回满足条件的 Q*，放大到概率量级
        return Q * n

# ============================================================================
# 2. 速率失真损失 (保持 O(d^3) 优化版)
# ============================================================================
class RateDistortionLoss(nn.Module):
    def __init__(self, emb_size, epsilon=0.5):
        super().__init__()
        self.d = emb_size
        self.eps_sq = epsilon 

    def calculate_rate(self, E, tr_pi=None, weighted_E=None):
        n = E.shape[0]
        identity = torch.eye(self.d, device=E.device)
        if weighted_E is not None:
            # 对应公式 (6) 的内部项
            matrix = identity + (self.d / (tr_pi * self.eps_sq)) * torch.mm(weighted_E.t(), weighted_E)
        else:
            # 对应公式 (5) 的内部项
            matrix = identity + (self.d / (n * self.eps_sq)) * torch.mm(E.t(), E)
        # 增加抖动保护稳定性
        return 0.5 * torch.logdet(matrix + identity * 1e-6)

    def forward(self, E, Q):
        """
        E: [N, d]
        Q: [N, K] 从 IPOT 获得的成员分配矩阵 (Pi)
        """
        n, K = Q.shape
        r_global = self.calculate_rate(E)
        
        r_cluster = 0.0
        # 计算每个簇的 Tr(Pi_k)
        tr_pi_all = torch.sum(Q, dim=0) + 1e-8
        
        # 遍历所有簇计算分簇速率 Rc (公式 6)
        for k in range(K):
            if tr_pi_all[k] < (n / K * 0.1): continue # 跳过几乎为空的簇
            q_k = Q[:, k].unsqueeze(1)
            weighted_E = E * torch.sqrt(q_k)
            r_cluster += (tr_pi_all[k] / (2 * n)) * self.calculate_rate(E, tr_pi_all[k], weighted_E)
            
        return r_cluster - r_global

# ============================================================================
# 3. nCL (Option II) 模型主类
# ============================================================================
class nCL(LightGCN):
    extra_log_args = ['alpha', 'num_clusters', 'temp', 'epsilon']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--alpha', type=float, default=0.01, help='平衡系数')
        parser.add_argument('--epsilon', type=float, default=0.1, help='失真常数')
        parser.add_argument('--num_clusters', type=int, default=300, help='聚类中心数 K')
        parser.add_argument('--temp', type=float, default=0.1, help='预测分温度系数')
        return LightGCN.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.alpha = args.alpha
        self.K = args.num_clusters
        self.temp = args.temp

        # 定义可学习的聚类中心 (对应论文中的 语义结构空间)
        self.user_centroids = nn.Parameter(torch.empty(self.K, self.emb_size))
        self.item_centroids = nn.Parameter(torch.empty(self.K, self.emb_size))
        nn.init.xavier_uniform_(self.user_centroids)
        nn.init.xavier_uniform_(self.item_centroids)

        # 实例化求解器
        self.ipot = IPOTSolver(self.K, iterations=20, beta=0.5)
        self.rd_loss = RateDistortionLoss(self.emb_size, args.epsilon)

    def forward(self, feed_dict):
        user, items = feed_dict['user_id'], feed_dict['item_id']
        u_embed, i_embed = self.encoder(user, items)
        
        # 归一化用于几何计算
        u_norm = F.normalize(u_embed, p=2, dim=-1, eps=1e-12)
        i_norm_all = F.normalize(i_embed, p=2, dim=-1, eps=1e-12)
        i_norm_pos = i_norm_all[:, 0, :]

        # 预测分 (余弦相似度 + 温度系数)
        prediction = (u_norm.unsqueeze(1) * i_norm_all).sum(dim=-1) / self.temp
        
        out_dict = {'prediction': prediction.reshape(feed_dict['batch_size'], -1)}

        if self.training:
            # 1. Alignment Loss (对齐)
            l_align = (u_norm - i_norm_pos).pow(2).sum(dim=1).mean()

            # 2. Compactness Loss (基于 IPOT 动态聚类)
            # --- 用户端 ---
            # 计算成本矩阵 C: 用户与中心的余弦相似度
            u_cost = torch.mm(u_norm, F.normalize(self.user_centroids, p=2, dim=-1).t())
            Q_u = self.ipot(u_cost) # 通过 IPOT 求解最优分配矩阵 Q*
            l_comp_u = self.rd_loss(u_norm, Q_u)

            # --- 物品端 ---
            i_flatten = i_norm_all.view(-1, self.emb_size)
            i_cost = torch.mm(i_flatten, F.normalize(self.item_centroids, p=2, dim=-1).t())
            Q_i = self.ipot(i_cost)
            l_comp_i = self.rd_loss(i_flatten, Q_i)

            l_comp = l_comp_u + l_comp_i
            
            out_dict.update({
                'loss': l_align + self.alpha * l_comp,
                'l_align': l_align.detach().item(),
                'l_comp': l_comp.detach().item()
            })
            
        return out_dict

    # def loss(self, out_dict):
    #     # 兼容 ReChorus BaseRunner
    #     return out_dict['loss'], out_dict['l_align'], out_dict['l_comp']