import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import to_dense_batch
from torch_geometric.data.batch import Batch

from torch_geometric.nn import  global_add_pool, global_mean_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import degree

from mamba_ssm import Mamba2

class MutualDTA(torch.nn.Module):
    def __init__(self, n_durg_features=512, n_protein_features=512, n_edge_features=6, protein_hidden_dim=256):

        super(MutualDTA, self).__init__()

        drug_hidden_dim = 128
        dropout_rate = 0.3 # dropout rate

        # DMPNN 
        dmpnn_hidden_dim = 64
        n_iter = 3
        
        # Mutual Attention & beta
        k, self.beta_p, self.beta_x  = 64, 0.5, 0.99

        # paramters for Mutual-Attention 
        self.W_b = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(protein_hidden_dim, drug_hidden_dim)))
        self.W_x = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, drug_hidden_dim)))
        self.W_p = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, protein_hidden_dim)))
        self.w_hx = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k,1)))
        self.w_hp = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k,1)))

        self.MLP_before_DMPNN = nn.Sequential(nn.Linear(n_durg_features, drug_hidden_dim), nn.LeakyReLU())
        self.dmpnn = DMPNN(drug_hidden_dim, n_edge_features, dmpnn_hidden_dim, n_iter)
        self.MLP_after_DMPNN = nn.Sequential(nn.Linear(dmpnn_hidden_dim, drug_hidden_dim),nn.LeakyReLU())
        
        # protein
        self.MLP_protein = nn.Sequential(nn.Linear(n_protein_features, protein_hidden_dim))
        self.mamba = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=protein_hidden_dim, # Model dimension d_model
            d_state=64,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )


        # MLP for concated durg and protein
        concat_feature_dim = drug_hidden_dim*2+protein_hidden_dim*2
        self.MLP_combined = nn.Sequential(
                                         nn.LayerNorm(concat_feature_dim,concat_feature_dim),
                                         nn.Linear(concat_feature_dim,concat_feature_dim),
                                         nn.LeakyReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.LayerNorm(concat_feature_dim),
                                         nn.Linear(concat_feature_dim, drug_hidden_dim),
                                         nn.LeakyReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(drug_hidden_dim, 1))
        

    def forward(self, durgs, proteins):
        # drugs: list of graph  proteins: tensor of [Batch, L, n_features]

        durgs = Batch.from_data_list(durgs, follow_batch=['edge_index'])
        durgs.x = self.MLP_before_DMPNN(durgs.x) # [n_atoms, n_features]
        x, mask = to_dense_batch(durgs.x, durgs.batch)
        x = F.dropout1d(x,training=self.training)

        durg_encoded = self.dmpnn(durgs)
        durg_encoded = (1-self.beta_x) * self.MLP_after_DMPNN(durg_encoded) + self.beta_x * durgs.x
        durg_encoded, mask = to_dense_batch(durg_encoded, durgs.batch)


        target = self.MLP_protein(proteins)
        target_mamba = (1-self.beta_p)*self.mamba(target) + self.beta_p*target

        durg, protein = self.mutual_attention(durg_encoded, target)
        
        drug_protein = torch.cat((x.mean(1),durg, target_mamba.mean(1),protein), -1)
        out = self.MLP_combined(drug_protein)
        
        return out

    def mutual_attention(self, x, target):  # durg : B x 128 x 45, target : B x L x 128

        x = x.permute(0, 2, 1)

        C = F.tanh(torch.matmul(target, torch.matmul(self.W_b, x))) # B x L x 45

        H_c = F.tanh(torch.matmul(self.W_x, x) + torch.matmul(torch.matmul(self.W_p, target.permute(0, 2, 1)), C))          # B x k x 45
        H_p = F.tanh(torch.matmul(self.W_p, target.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_x, x), C.permute(0, 2, 1)))# B x k x L

        a_c_weight = torch.matmul(torch.t(self.w_hx), H_c)
        a_p_weight = torch.matmul(torch.t(self.w_hp), H_p)


        a_c = F.softmax(a_c_weight, dim=2) # B x 1 x 45
        a_p = F.softmax(a_p_weight, dim=2) # B x 1 x L

        c = torch.squeeze(torch.matmul(a_c, x.permute(0, 2, 1)))      # B x 128
        p = torch.squeeze(torch.matmul(a_p, target))                  # B x 128

        return c, p
    



class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)

        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx

# Substructure attention mechanism
class DMPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, n_feats, n_iter):
        super().__init__()
        self.n_iter = n_iter
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, n_feats),
            nn.LeakyReLU(),
        )
        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)

        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))
        glorot(self.a)
        self.lin_block = LinearBlock(n_feats)
        self.scale = n_feats**(-0.5)
        self.w = nn.Parameter(torch.zeros((1,n_feats,1)))
        nn.init.xavier_normal_(self.w)

    def forward(self, data):

        x_in = self.mlp(data.x)
        edge_index = data.edge_index
        edge_u = self.lin_u(x_in)
        edge_v = self.lin_v(x_in)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv)/3
        out = edge_attr
        
        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + out
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        scores = torch.matmul(gout_all.permute(0,2,1), self.w).permute(0,2,1) * self.scale
        scores = torch.softmax(scores, dim=-1)
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)

        out = (out_all * scores).sum(-1)

        x = x_in + scatter(out , edge_index[1], dim_size=x_in.size(0), dim=0, reduce='add')

        return x
    
   

    

    
class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = n_feats//2
        self.gn1 = GraphNorm(n_feats)
        self.lin1 = nn.Sequential(
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.gn2 = GraphNorm(self.snd_n_feats)
        self.lin2 = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.gn3 = GraphNorm(self.snd_n_feats)
        self.lin3 = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.gn4 = GraphNorm(self.snd_n_feats)
        self.lin4 = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.gn5 = GraphNorm(self.snd_n_feats)
        self.lin5 = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x, batch):
        x = self.lin1(self.gn1(x,batch))
        x = (self.lin3(self.gn3(self.lin2(self.gn2(x,batch)))) + x) / 2
        x = (self.lin4(self.gn4(x,batch)) + x) / 2
        x = self.lin5(self.gn5(x,batch))
        return x   