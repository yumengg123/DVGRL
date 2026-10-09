
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, device):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, h_i, h_j, weight=None):
        """
        无batch版本的对比损失
        h_i: [num_spots, feature_dim] - 视图1的节点嵌入
        h_j: [num_spots, feature_dim] - 视图2/融合的节点嵌入
        """
        num_spots = h_i.shape[0]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature  # [num_spots, num_spots]
        
        # 提取正样本对的相似度（对角线元素）
        positives = torch.diag(similarity_matrix)  # [num_spots]
        
        # 创建掩码矩阵，用于排除正样本对
        mask = torch.ones((num_spots, num_spots)).to(self.device)
        mask = mask.fill_diagonal_(0)  # 将对角线置为0
        
        # 计算对比损失
        nominator = torch.exp(positives)  # [num_spots]
        denominator = mask * torch.exp(similarity_matrix)  # [num_spots, num_spots]
        denominator_sum = torch.sum(denominator, dim=1)  # [num_spots]
        
        # 避免数值不稳定，添加小常数
        denominator_sum = torch.clamp(denominator_sum, min=1e-8)
        
        loss_partial = -torch.log(nominator / denominator_sum)  # [num_spots]
        loss = torch.mean(loss_partial)  # 标量
        
        # 如果提供了权重，则对损失进行加权
        loss = weight * loss if weight is not None else loss

        return loss
'''
'''
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, h_i, h_j, weight=None):
        """
        h_i, h_j: [B, N, D]
        """
        B, N, D = h_i.shape

        h_i = F.normalize(h_i, dim=-1)
        h_j = F.normalize(h_j, dim=-1)

        # 相似度矩阵
        sim = torch.matmul(h_i, h_j.transpose(-2, -1))  # [B, N, N]
        sim = sim / self.temperature

        # 数值稳定
        sim = sim - sim.max(dim=-1, keepdim=True)[0]

        # 正样本
        pos = torch.exp(torch.diagonal(sim, dim1=-2, dim2=-1))  # [B, N]

        # 分母（包含正样本）
        denom = torch.exp(sim).sum(dim=-1)  # [B, N]

        loss = -torch.log(pos / denom)  # [B, N]

        loss = loss.mean()

        if weight is not None:
            loss = weight * loss

        return loss
'''
'''
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, device, batch_size):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size

    def forward(self, h_i, h_j, weight=None):
      
        batch_size, num_nodes, feature_dim = h_i.shape
        
        similarity_matrix = torch.matmul(h_i, h_j.transpose(1, 2)) / self.temperature
     
        positives = torch.diagonal(similarity_matrix, dim1=1, dim2=2)
   
        mask = torch.ones((batch_size, num_nodes, num_nodes), device=self.device)
        mask = mask - torch.eye(num_nodes, device=self.device).unsqueeze(0)
        
        exp_similarity = torch.exp(similarity_matrix)
        denominator = torch.sum(mask * exp_similarity, dim=2)
    
        denominator = torch.clamp(denominator, min=1e-8)
      
        loss_per_node = -torch.log(torch.exp(positives) / denominator)
      
        loss_per_batch = torch.mean(loss_per_node, dim=1)
        
        total_loss = torch.mean(loss_per_batch)
      
        if weight is not None:
            total_loss = weight * total_loss
        
        return total_loss
'''
'''
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, device,batch_size):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size

    def forward(self, h_i, h_j, weight=None):
        """
        在每个batch内计算对比损失
        h_i: [batch_size, num_nodes, feature_dim] - 视图1的节点嵌入
        h_j: [batch_size, num_nodes, feature_dim] - 视图2/融合的节点嵌入
        """
        batch_size, num_nodes, feature_dim = h_i.shape
        
        total_loss = 0
        
        # 对每个batch单独计算对比损失
        for b in range(batch_size):
            # 获取当前batch的节点嵌入
            h_i_batch = h_i[b]  # [num_nodes, feature_dim]
            h_j_batch = h_j[b]  # [num_nodes, feature_dim]
            
            # 计算相似度矩阵
            similarity_matrix = torch.matmul(h_i_batch, h_j_batch.T) / self.temperature  # [num_nodes, num_nodes]
            
            # 提取正样本对的相似度（对角线元素）
            positives = torch.diag(similarity_matrix)  # [num_nodes]
            
            # 创建掩码矩阵，用于排除正样本对
            mask = torch.ones((num_nodes, num_nodes)).to(self.device)
            mask = mask.fill_diagonal_(0)  # 将对角线置为0
            
            # 计算对比损失
            nominator = torch.exp(positives)  # [num_nodes]
            denominator = mask * torch.exp(similarity_matrix)  # [num_nodes, num_nodes]
            denominator_sum = torch.sum(denominator, dim=1)  # [num_nodes]
            
            # 避免数值不稳定，添加小常数
            denominator_sum = torch.clamp(denominator_sum, min=1e-8)
            
            loss_partial = -torch.log(nominator / denominator_sum)  # [num_nodes]
            batch_loss = torch.mean(loss_partial)  # 当前batch的损失
            
            total_loss += batch_loss
        
        # 计算所有batch的平均损失
        total_loss = total_loss / batch_size
        
        # 如果提供了权重，则对损失进行加权
        total_loss = weight * total_loss if weight is not None else total_loss

        return total_loss
'''
