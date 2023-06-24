import torch
from torch import nn


class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size, embed_size, block_size, dropout=0.):
        super().__init__()
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        #compute attn. scores
        weight = q @ k.transpose(-1,-2) * C**-0.5 # (B,T,T)
        #mask out upper half of the scores since this is decoder
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        #normalize to [0,1]
        weight = torch.softmax(weight, dim=-1)  # (B,T,T)
        weight = self.dropout(weight) # (B,T,T)
        v = self.value(x) # (B,T,C)
        #apply attention scores to values
        output = weight @ v # (B,T,C)
        return output
    

class MultiHeadAttention(nn.Module):
    '''combines multiple self-attention heads with a linear projection'''
    def __init__(self, num_heads, head_size, embed_size, block_size, dropout=0.):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embed_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,C)
        out = self.proj(out) # (B,T,C)
        out = self.dropout(out) # (B,T,C)
        return out
    

class MLP(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.ReLU(),
            nn.Linear(embed_size*4, embed_size),
            nn.Dropout(dropout)
            )
        
    def forward(self, x):
        return self.layers(x)
    


class Block(nn.Module):
    '''transformer block'''
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        head_size = embed_size // num_heads
        self.attn = MultiHeadAttention(num_heads, head_size, embed_size, block_size, dropout)
        self.mlp = MLP(embed_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
