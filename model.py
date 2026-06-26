
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl import function as fn

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

import  dgl

class MLPModel(nn.Module):
    def __init__(self, args, k_token, input_dim, hidden_dim, num_classes, dropout, k1):
        super(MLPModel, self).__init__()
        
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ffn_dim = args.ffn_dim

        self.token_wise_mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) for _ in range(k_token)]
        )
        
        ###自由可学习的权重矩阵W
        self.W_k = nn.Parameter(torch.Tensor(k_token, k1))
        nn.init.xavier_uniform_(self.W_k)
        
        # self.W_k = torch.randn(k_token, k_token) # 完全随机的固定matrix
        
        # self.W_k = torch.eye(k_token, k1)

        self.residual_proj = nn.Linear(k_token, k1) if k_token != k1 else nn.Identity()
        
        self.k1_weights = nn.Parameter(torch.ones(k1) / k1)
        # self.k1_weights = torch.ones(k1) / k1
        
        self.FFN = nn.Sequential(
            nn.Linear(hidden_dim, self.ffn_dim),
            nn.LayerNorm(self.ffn_dim),
            nn.ReLU(),
            nn.Dropout(args.attention_dropout)
        )
        self.classifier = nn.Sequential(
                nn.Linear(self.ffn_dim, num_classes),
                # nn.Linear(args.hidden_dim, num_classes),
                nn.ReLU(),
                nn.Dropout(dropout),
            )     
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        self.mix_activation = args.activation
        
        self.belta = args.belta
        self.dataset = args.dataset
    
    def apply_mix_activation(self, x):
        if self.mix_activation == 'softmax':
            return F.softmax(x, dim=0)
        elif self.mix_activation == 'tanh':
            return torch.tanh(x)
        elif self.mix_activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.mix_activation == 'relu':
            return F.relu(x)
        else:
            raise ValueError(f"Unsupporting Activation: {self.mix_activation}")

    def forward(self, x): 
        # x shape: [batch_size, k_token, input_dim]
        # [N,K,hidden]
        d = x.shape[-1]//2
        x_origin = x[0, :d, :]
        x = torch.stack([layer(x[:,idx,:]) for idx, layer in enumerate(self.token_wise_mlp)], dim=1)
        
        # x = self.token_mlp(x)

        K_K1_mix = self.apply_mix_activation(self.W_k)  
        K_K1_mix = K_K1_mix.to(x.device)
        x_mixed = torch.matmul(x.transpose(1,2), K_K1_mix).transpose(1,2) # [N,hidden,K]@[K,k1] = [N,hidden,k1]   ->   [N,k1,hidden]

        residual = self.residual_proj(x.transpose(1,2)).transpose(1,2)
        alpha = torch.sigmoid(self.residual_weight)
        
        # x_mixed = x
        
        x_mixed = alpha * x_mixed + (1-alpha) * residual
        x = self.norm(x_mixed)
        x = self.act(x)
        ## k1->1
        weights = F.softmax(self.k1_weights, dim=0) # [k']
        weights = weights.to(x.device)

        x_agg = torch.einsum('nkh,k->nh', x, weights)

        x_agg = self.FFN(x_agg)
        # ablation
        # x_agg = self.norm2(x_agg)
        # if self.dataset in ['wisconsin', 'cornell', 'texas']:
        #     x_agg = x_agg + self.belta * x_origin
        
        return F.log_softmax(self.classifier(x_agg), dim=1) #, K_K1_mix.cpu()
