import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from models.general.LightGCN import *
from tqdm import tqdm

# 1. 辅助工具：预计算图拓扑共现邻居 使用图方法计算邻居从而计算局部紧凑性损失

def get_nclg_neighbors(train_mat, user_count, item_count, topk=50, chunk_size=1000):
    print(f"nCLG: Pre-computing neighbors (Dataset scale: {user_count} users)...")
    
    rows, cols = [], []
    for u, items in train_mat.items():
        for i in items:
            rows.append(u)
            cols.append(i)
    R = sp.csr_matrix(([1.0]*len(rows), (rows, cols)), shape=(user_count, item_count))
    #用户端
    item_freq = np.array(R.sum(axis=0)).flatten()
    idf = np.log(user_count / (1 + item_freq) + 1.0)
    R_u_final = R.dot(sp.diags(np.sqrt(idf)))
    u_norms = np.array(np.sqrt(R_u_final.multiply(R_u_final).sum(axis=1))).flatten()
    R_u_final = sp.diags(1.0 / (u_norms + 1e-12)).dot(R_u_final).tocsr()

    # 物品端
    user_freq = np.array(R.sum(axis=1)).flatten()
    u_idf = np.log(item_count / (1 + user_freq) + 1.0)
    R_i_final = sp.diags(np.sqrt(u_idf)).dot(R)
    i_norms = np.array(np.sqrt(R_i_final.multiply(R_i_final).sum(axis=0))).flatten()
    R_i_final = R_i_final.dot(sp.diags(1.0 / (i_norms + 1e-12))).tocsc().transpose().tocsr()

    def chunked_topk(mat, size, k):
        neighbors = torch.zeros((size, k), dtype=torch.long)
        for i in range(0, size, chunk_size):
            end = min(i + chunk_size, size)
            batch_adj = mat[i:end].dot(mat.transpose()).toarray()
            for r in range(batch_adj.shape[0]):
                batch_adj[r, i + r] = 0 
            topk_idx = np.argpartition(batch_adj, -k, axis=1)[:, -k:]
            neighbors[i:end] = torch.from_numpy(topk_idx)
        return neighbors

    u_nb = chunked_topk(R_u_final, user_count, topk)
    i_nb = chunked_topk(R_i_final, item_count, topk)
    return u_nb, i_nb

# 2. 支持注意力权重的速率失真损失 紧凑性损失计算类

class RateDistortionLoss(torch.nn.Module):
    def __init__(self, emb_size, epsilon=0.5):
        super(RateDistortionLoss, self).__init__()
        self.d = emb_size
        self.eps_sq = epsilon 

    #根据公式计算R
    def calculate_rate(self, E):
        n, d = E.shape
        identity = torch.eye(d, device=E.device)
        # 增加 1e-8 稳定性
        matrix = identity + (d / (n * self.eps_sq + 1e-8)) * torch.mm(E.t(), E)
        matrix = matrix + identity * 1e-6
        return 0.5 * torch.logdet(matrix)

    def forward(self, E, cluster_embs, weights=None):
        # 全局紧凑性损失
        r_global = self.calculate_rate(E)
        
        # 加权局部压缩
        if weights is not None:
            # 实现 Cov_weighted = sum(w * e * e^T)
            # 通过 e_new = e * sqrt(w) 实现向量化
            weighted_embs = cluster_embs * torch.sqrt(weights.unsqueeze(-1) + 1e-9)
        else:
            weighted_embs = cluster_embs

        # 批量计算协方差 [S, d, d]
        cluster_covs = torch.bmm(weighted_embs.transpose(1, 2), weighted_embs)
        
        S, K, D = cluster_embs.shape
        identity = torch.eye(D, device=E.device).unsqueeze(0)
        scale = self.d / (K * self.eps_sq + 1e-8)
        
        matrices = identity + scale * cluster_covs
        matrices = matrices + identity * 1e-6
        
        _, logdets = torch.linalg.slogdet(matrices)
        # 加权平均
        r_cluster = (K / (2 * E.shape[0])) * logdets.sum()
        
        return r_cluster - r_global

# 3. 主模型：nCLG 
class nCLG(LightGCN):
    extra_log_args = ['alpha', 'epsilon', 'temp', 'num_seeds', 'neighbor_topk', 'item_multi']
    
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--alpha', type=float, default=0.01)
        parser.add_argument('--epsilon', type=float, default=0.3)
        parser.add_argument('--temp', type=float, default=0.2)
        parser.add_argument('--num_seeds', type=int, default=20)
        parser.add_argument('--neighbor_topk', type=int, default=20)
        parser.add_argument('--item_multi', type=float, default=1.0, help='Multiplier for Item expansion')
        return LightGCN.parse_model_args(parser)

    def __init__(self, args, corpus):
        super(nCLG, self).__init__(args, corpus)
        
        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.temp = args.temp
        self.num_seeds = args.num_seeds
        self.neighbor_topk = args.neighbor_topk
        self.item_multi = args.item_multi

        self.rd_loss = RateDistortionLoss(self.emb_size, self.epsilon)
        # 预计算邻居
        u_nb, i_nb = get_nclg_neighbors(corpus.train_clicked_set, self.user_num, self.item_num, self.neighbor_topk)
        self.register_buffer('u_neighbors_tensor', u_nb)
        self.register_buffer('i_neighbors_tensor', i_nb)

    def get_attentive_compactness(self, all_embeddings, neighbors_table, batch_indices):
        """计算带注意力的邻域紧凑性"""
        if len(batch_indices) > self.num_seeds:
            rand_idx = torch.randperm(len(batch_indices), device=batch_indices.device)[:self.num_seeds]
            seed_indices = batch_indices[rand_idx]
        else:
            seed_indices = batch_indices
            
        # 提取数据
        seed_embs = all_embeddings[seed_indices] # [S, d]
        nb_indices = neighbors_table[seed_indices] # [S, K]
        nb_embs = all_embeddings[nb_indices] # [S, K, d]
        
        # 归一化
        seed_norm = F.normalize(seed_embs, p=2, dim=-1)
        nb_norm = F.normalize(nb_embs, p=2, dim=-1)
        batch_norm_E = F.normalize(all_embeddings[batch_indices], p=2, dim=-1)
        
        # --- 注意力机制 ---
        # 计算种子与邻居的相似度 [S, 1, d] * [S, d, K] -> [S, K]
        attn_scores = torch.bmm(seed_norm.unsqueeze(1), nb_norm.transpose(1, 2)).squeeze(1)
        # 用 temp 锐化分布
        attn_weights = F.softmax(attn_scores / self.temp, dim=-1)
        # 对齐能量：让权重均值为 1
        attn_weights = attn_weights * self.neighbor_topk
        
        return self.rd_loss(batch_norm_E, nb_norm, weights=attn_weights)

    def forward(self, feed_dict):
        user, items = feed_dict['user_id'], feed_dict['item_id']
        all_u_embed, all_i_embed = self.encoder.get_all_embeddings() 
        #对齐性损失
        u_norm = F.normalize(all_u_embed[user], p=2, dim=-1)
        i_norm_all = F.normalize(all_i_embed[items], p=2, dim=-1)
        i_norm_pos = i_norm_all[:, 0, :]
        l_align = (u_norm - i_norm_pos).pow(2).sum(dim=1).mean()
        #紧凑性损失
        if self.training:
            l_comp_u = self.get_attentive_compactness(all_u_embed, self.u_neighbors_tensor, user)
            unique_items = items.view(-1).unique()
            l_comp_i = self.get_attentive_compactness(all_i_embed, self.i_neighbors_tensor, unique_items)
            l_comp = l_comp_u + self.item_multi * l_comp_i
        else:
            l_comp = torch.tensor(0.0, device=u_norm.device)
        
        prediction = (u_norm.unsqueeze(1) * i_norm_all).sum(dim=-1) / self.temp
        
        loss = l_align + self.alpha * l_comp
        
        return {
            'loss': loss, 
            'prediction': prediction.reshape(feed_dict['batch_size'], -1),
            'l_align': l_align.detach().item() if self.training else 0.0, 
            'l_comp': l_comp.detach().item() if self.training else 0.0
        }