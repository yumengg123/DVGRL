import torch
from .preprocess import compute_ppr, preprocess_diff, preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from .model import Model
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from .loss import ContrastiveLoss
import pandas as pd
import matplotlib.pyplot as plt
import os

class MVGRLST():
    def __init__(self, 
        adata,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00, 
        random_seed = 41,
        alpha = 1,
        beta =4,  
        dim_input=3000,
        dim_output=512,
        pre_epochs=300,
        con_epochs=100,
        sample_size = 2000,
        batch_size = 4,
        sparse = False,
        temperature=0.1,
        #alpha_1=20,       
        lambda_cl=0.15,
        num_views=2,
        #ablation_mode='full' 
        ):
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.pre_epochs = pre_epochs
        self.con_epochs = con_epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.sparse = sparse
        self.temperature = temperature
        self.lambda_cl = lambda_cl
        self.num_views = num_views
        #self.alpha_1= alpha_1
        # 初始化自适应对比损失
        
        self.adaptive_contrastive_loss = ContrastiveLoss(
            temperature=self.temperature,
            device=self.device,
            batch_size=self.batch_size
        )
        '''
        self.adaptive_contrastive_loss = ContrastiveLoss(
            temperature=self.temperature,
            device=self.device,
           # batch_size=self.batch_size
        )
        '''
        #self.ablation_mode = ablation_mode#消融实验
        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)
        
        if 'adj' not in adata.obsm.keys(): 
            construct_interaction(self.adata)
            compute_ppr(self.adata, 0.8)     #修改
           
        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.diff = self.adata.obsm['diff']
        #self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        self.diff = preprocess_diff(self.diff, self.adj)
        self.diff = torch.FloatTensor(self.diff).to(self.device)
        self.adj = preprocess_adj(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)
        
        self.lbl_1 = torch.ones(self.batch_size, self.sample_size * 2)
        self.lbl_2 = torch.zeros(self.batch_size, self.sample_size * 2)
        self.lbl = torch.cat((self.lbl_1, self.lbl_2), 1)
        self.lbl = torch.FloatTensor(self.lbl).to(self.device)
    
    def compute_view_weights(self, z_views, z_fused):
        batch_size, sample_size, dim = z_fused.shape
        #
        w = []
        
        for b in range(batch_size):
            
            z_fused_batch = z_fused[b]  
            w_batch = []
           
            global_sim = torch.matmul(z_fused_batch, z_fused_batch.t())  
            for v in range(len(z_views)):
                
                z_view_batch = z_views[v][b] 

                view_sim = torch.matmul(z_view_batch, z_view_batch.t())  
                
                related_sim = torch.matmul(z_view_batch, z_fused_batch.t()) 
            
                 
                w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (sample_size * sample_size)
                w_batch.append(torch.exp(-w_v))
            w.append(torch.stack(w_batch))
        
        w = torch.stack(w)  
        w = torch.mean(w, dim=0)  
        w = w / torch.sum(w) 
    
        return w
    '''
    def compute_view_weights(self, z_views, z_fused):
               
        
        batch_size, sample_size, dim = z_fused.shape
        num_views = len(z_views)
   
        z_views_stack = torch.stack(z_views, dim=0)
    
        global_sim = torch.matmul(z_fused, z_fused.transpose(1, 2))
        view_sim = torch.matmul(z_views_stack, z_views_stack.transpose(2, 3))
        related_sim = torch.matmul(z_views_stack, z_fused.transpose(1, 2))
    
        sum_view_sim = torch.sum(view_sim, dim=(2, 3)) 
        sum_global_sim = torch.sum(global_sim, dim=(1, 2))  
        sum_related_sim = torch.sum(related_sim, dim=(2, 3))  
    
   
        sum_global_sim_expanded = sum_global_sim.unsqueeze(0).expand(num_views, -1)
   
        w_v = (sum_view_sim + sum_global_sim_expanded - 2 * sum_related_sim) / (sample_size * sample_size)
    
        w = torch.exp(-w_v)
        w = torch.mean(w, dim=1)
        w = w / torch.sum(w)
    
        return w
    '''
    def pretrain_phase(self):
        """
        预训练阶段 - 仅使用对比学习损失和重构损失
        """
        print('开始预训练阶段...')
        self.model.train()
        
        for epoch in tqdm(range(self.pre_epochs)): 
            self.model.train()
            # 随机采样
            self.idx = np.random.randint(0, self.adj.shape[-1] - self.sample_size + 1, self.batch_size)
            self.ba, self.bd, self.bf = [], [], []
            #self.attention_weights = []
            for i in self.idx:
                self.ba.append(self.adj[i: i + self.sample_size, i: i + self.sample_size])
                self.bd.append(self.diff[i: i + self.sample_size, i: i + self.sample_size])
                self.bf.append(self.features[i: i + self.sample_size])

            self.ba = torch.stack(self.ba).reshape(self.batch_size, self.sample_size, self.sample_size)#批次邻接矩阵
            self.bd = torch.stack(self.bd).reshape(self.batch_size, self.sample_size, self.sample_size)#批次扩散矩阵
            self.bf = torch.stack(self.bf).reshape(self.batch_size, self.sample_size, self.dim_input)#批次特征矩阵
              
            # 特征置换用于对比学习
            self.ids = np.random.permutation(self.sample_size)#随机置换索引，用于生成对比学习的负样本
            self.shuf_fts = self.bf[:, self.ids, :]#形状同self.bf,通过打乱节点顺序创建"负视图"，用于对比学习
            
            self.ret, self.z1_norm, self.z2_norm, self.recon_h1, self.recon_h2,self.z_fused,self.c_1,self.c_2,self.z_fused_normal = self.model(self.bf, self.shuf_fts, self.ba, self.bd, self.sparse, None, None, None)
            #self.ret, self.z1_norm, self.z2_norm, self.recon_h1, self.recon_h2,  self.recon_h3,self.z_fused,self.c_1,self.c_2,self.z_fused_normal = self.model(self.bf, self.shuf_fts, self.ba, self.bd, self.sparse, None, None, None)
            '''
            self.ret, self.z1_norm, self.z2_norm, self.recon_h1, self.recon_h2, self.z_fused,self.c_1,self.c_2,self.z_fused_gobal = self.model(
                self.bf, self.shuf_fts, self.ba, self.bd, self.sparse, None, None, None)
            '''
            # 计算损失 - 预训练阶段不使用自适应对比损失
            self.loss_sl = self.loss_CSL(self.ret, self.lbl)
            #self.loss_sl_1 = self.loss_CSL(self.ret, self.lbl)
            #self.loss_sl_2 = self.loss_CSL(self.ret_a, self.lbl)
            self.loss_feat_1 = F.mse_loss(self.bf, self.recon_h1)
            self.loss_feat_2 = F.mse_loss(self.bf, self.recon_h2)
            # 预训练阶段损失组合
            loss = (self.alpha * self.loss_sl + self.beta * (self.loss_feat_1 + self.loss_feat_2))
            #loss = self.alpha * (self.loss_sl_1+self.loss_sl_2) + self.beta * (self.loss_feat_1 + self.loss_feat_2)
            # 记录损失历史
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            
        print("预训练阶段完成!")
        
    def adaptive_contrastive_train_phase(self):
        """
        自适应对比训练阶段 - 添加自适应对比损失
        """
        print('开始自适应对比训练阶段...')
        self.model.train()
        
        for epoch in tqdm(range(self.con_epochs)): 
            self.model.train()
            # 随机采样
            self.idx = np.random.randint(0, self.adj.shape[-1] - self.sample_size + 1, self.batch_size)
            self.ba, self.bd, self.bf = [], [], []
            self.attention_weights = []
            for i in self.idx:
                self.ba.append(self.adj[i: i + self.sample_size, i: i + self.sample_size])
                self.bd.append(self.diff[i: i + self.sample_size, i: i + self.sample_size])
                self.bf.append(self.features[i: i + self.sample_size])

            self.ba = torch.stack(self.ba).reshape(self.batch_size, self.sample_size, self.sample_size)
            self.bd = torch.stack(self.bd).reshape(self.batch_size, self.sample_size, self.sample_size)
            self.bf = torch.stack(self.bf).reshape(self.batch_size, self.sample_size, self.dim_input)
              
            # 特征置换用于对比学习
            self.ids = np.random.permutation(self.sample_size)
            self.shuf_fts = self.bf[:, self.ids, :]
            self.ret, self.z1_norm, self.z2_norm, self.recon_h1, self.recon_h2,self.z_fused,self.c_1,self.c_2,self.z_fused_normal = self.model(self.bf, self.shuf_fts, self.ba, self.bd, self.sparse, None, None, None)
            #self.ret, self.z1_norm, self.z2_norm, self.recon_h1, self.recon_h2,  self.recon_h3,self.z_fused,self.c_1,self.c_2,self.z_fused_normal = self.model(self.bf, self.shuf_fts, self.ba, self.bd, self.sparse, None, None, None)
            #self.ret,self.ret_a, self.z1_norm, self.z2_norm, self.recon_h1, self.recon_h2, self.z_fused,self.c_1,self.c_2,self.z_fused_gobal= self.model(
                #self.bf, self.shuf_fts, self.ba, self.bd, self.sparse, None, None, None)
            #self.ret, self.z1_norm, self.z2_norm, self.recon_h1, self.recon_h2,  self.recon_h3,self.z_fused,self.c_1,self.c_2,self.z_fused_normal = self.model(
                #self.bf, self.shuf_fts, self.ba, self.bd, self.sparse, None, None, None)
            # 计算基础损失
            self.loss_sl = self.loss_CSL(self.ret, self.lbl)
            #self.loss_sl_1 = self.loss_CSL(self.ret, self.lbl)
            #self.loss_sl_2 = self.loss_CSL(self.ret_a, self.lbl)
            self.loss_feat_1 = F.mse_loss(self.bf, self.recon_h1)
            self.loss_feat_2 = F.mse_loss(self.bf, self.recon_h2)
            #self.loss_feat_3 = F.mse_loss(self.bf, self.recon_h3)
            # 计算自适应对比损失（无梯度计算）
            with torch.no_grad():
                # 计算视图权重
                view_weights = self.compute_view_weights([self.z1_norm, self.z2_norm], self.z_fused)
                #view_weights = self.compute_view_weights([self.z1_norm.mean(dim=1), self.z2_norm.mean(dim=1)],self.z_fused_normal.mean(dim=1))
                
           # loss_contrastive_1=self.adaptive_contrastive_loss(self.c_1, self.z_fused_gobal, weight=view_weights[0] if len(view_weights) > 0 else 1.0)
            #loss_contrastive_2 = self.adaptive_contrastive_loss(self.c_2, self.z_fused_gobal, weight=view_weights[1] if len(view_weights) > 1 else 1.0)
            ''' 
            loss_contrastive_1=self.adaptive_contrastive_loss(
            self.z1_norm.mean(dim=1), self.z_fused.mean(dim=1), weight=view_weights[0] if len(view_weights) > 0 else 1.0
        )
            loss_contrastive_2 = self.adaptive_contrastive_loss(
            self.z2_norm.mean(dim=1), self.z_fused.mean(dim=1), weight=view_weights[1] if len(view_weights) > 1 else 1.0
        )
            '''
            loss_contrastive_1=self.adaptive_contrastive_loss(
            self.z1_norm, self.z_fused, weight=view_weights[0] if len(view_weights) > 0 else 1.0
        )
            loss_contrastive_2 = self.adaptive_contrastive_loss(
            self.z2_norm, self.z_fused, weight=view_weights[1] if len(view_weights) > 1 else 1.0
        )
            self.loss_adaptive_contrastive = loss_contrastive_1 + loss_contrastive_2
            # 自适应对比训练阶段损失组合
            #loss = (self.alpha * self.loss_sl + 
                   #self.beta * (self.loss_feat_1 + self.loss_feat_2+3*self.loss_feat_3) + 
                   #self.lambda_cl * self.loss_adaptive_contrastive)
            loss = (self.alpha * (self.loss_sl) + self.beta * (self.loss_feat_1 + self.loss_feat_2) + self.lambda_cl * self.loss_adaptive_contrastive)

            
            # 打印损失详情（可选）
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Adaptive weights - View1: {view_weights[0]:.4f}, View2: {view_weights[1]:.4f}")
                print(f"Adaptive contrastive loss: {self.loss_adaptive_contrastive.item():.6f}")
            
            # 记录损失历史
        
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            
        print("自适应对比训练阶段完成!")
       
    def train(self):
        """
        完整的训练流程：预训练 + 自适应对比训练
        """
        # 初始化模型和优化器
        self.model = Model(self.dim_input, self.dim_output, num_views=self.num_views).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        # 阶段1: 预训练
        self.pretrain_phase()
        
        # 阶段2: 自适应对比训练
        self.adaptive_contrastive_train_phase()
        
        # 生成最终嵌入
        self._generate_embeddings()
        return self.adata
    def _generate_embeddings(self):
        
        # 调整维度
        self.features = self.features.unsqueeze(0)
        self.adj = self.adj.unsqueeze(0)
        self.diff = self.diff.unsqueeze(0)
  
       # with torch.no_grad():
            #self.model.eval()
            #self.emb_rec = self.model.embed(self.features, self.adj, self.diff, self.sparse, None)[0].squeeze().detach().cpu().numpy()
            #self.adata.obsm['emb'] = self.emb_rec
        with torch.no_grad():
            
            self.model.eval()
        # 统一使用 embed
            emb, beta = self.model.embed(self.features, self.adj, self.diff, self.sparse, None)
        
        # 保存嵌入
            emb = emb.squeeze(0).cpu().numpy()
            self.adata.obsm['emb'] = emb
        
       
        
       
            