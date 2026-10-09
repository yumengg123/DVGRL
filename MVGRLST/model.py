import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.functional import normalize

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return out
#对节点进行注意力计算
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=2)
        return (beta * z).sum(2), beta

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits

class Decoder(nn.Module):
    def __init__(self, n_h, n_out):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(n_h, n_out)

    def forward(self, x):
        return self.fc(x)    

class Model(nn.Module):
    def __init__(self, n_in, n_h,num_views=2):
        super(Model, self).__init__()
        self.num_views = num_views
        
        # 为每个视图创建GCN编码器
        self.gcns = nn.ModuleList([GCN(n_in, n_h) for _ in range(num_views)])
        
        self.read = Readout()
        self.act = nn.PReLU()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        #self.graph_neigh = graph_neigh
        
        self.decoders = nn.ModuleList([Decoder(n_h, n_in) for _ in range(num_views)])
        self.decoder_fused = Decoder(n_h, n_in) 
        self.attention = Attention(n_h)
        #self.attention_view = Attention_view(n_h)
        

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
       
        h_1 = self.gcns[0](seq1, adj, sparse)
        emb_1 = self.act(h_1)
        #c_1 = self.read(emb_1, self.graph_neigh)
        c_1 = self.read(emb_1, msk)
        c_1 = self.sigm(c_1)
        
        
        h_2 = self.gcns[1](seq1, diff, sparse)#[batch_size, sample_size, n_h]
        emb_2 = self.act(h_2)
        c_2 = self.read(emb_2, msk)
        c_2 = self.sigm(c_2)
        
        h_3 = self.gcns[0](seq2, adj, sparse)
        emb_3 = self.act(h_3)
        c_3 = self.read(emb_3, msk)
        #c_3 = self.read(emb_3, self.graph_neigh)
        c_3 = self.sigm(c_3)
        
        h_4 = self.gcns[1](seq2, diff, sparse)
        emb_4 = self.act(h_4)
        c_4 = self.read(emb_4, msk)
        c_4 = self.sigm(c_4)
        
        z1_norm = F.elu(F.normalize(emb_1, p=2, dim=2))
        z2_norm = F.elu(F.normalize(emb_2, p=2, dim=2))
    
        
        ret = self.disc(c_1, c_2, emb_1, emb_2, emb_3, emb_4, samp_bias1, samp_bias2)
        
        z = torch.stack([z1_norm, z2_norm], dim=2)
        z_fused,_= self.attention(z)
        z_fused_1 = self.act(z_fused)
        z_fused_normal = F.normalize(z_fused_1, p=2, dim=2)
        #z_fused_gobal = self.read(z_fused_1, msk)
        
        # 重构特征
        recon_h1 = self.decoders[0](h_1)
        recon_h2 = self.decoders[1](h_2)
        #新添加
        recon_h3 = self.decoder_fused(z_fused)
        #return ret, z1_norm, z2_norm, recon_h1, recon_h2, z_fused
        return ret,z1_norm, z2_norm, recon_h1, recon_h2,z_fused,c_1,c_2,z_fused_normal

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcns[0](seq, adj, sparse)
        h_2 = self.gcns[1](seq, diff, sparse)
        
        
        emb = torch.stack([h_1, h_2], dim=2)
        emb, beta = self.attention(emb)
        return emb,beta
        #emb = torch.cat((h_1, h_2), dim=2)
        #return emb
        #return (h_1 + h_2).detach()
