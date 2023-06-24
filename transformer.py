import torch
from torch import nn
from attention import Block
import torch.nn.functional as F

class TransformerLanguageModel(nn.Module):
    '''transformer bigram language model'''

    def __init__(self, vocab_size, embed_size, block_size, num_heads, num_layers, dropout):
        super().__init__()
        #encode token id
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)

        #encode token position within each block
        self.position_embedding_table = nn.Embedding(block_size, embed_size)

        #transformer layers.
        self.layers = nn.Sequential(*[Block(embed_size, 
                                            num_heads=num_heads, 
                                            block_size=block_size, 
                                            dropout=dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size) # final layer norm
        self.lm_head = nn.Linear(embed_size, vocab_size) #final linear layer
        self.block_size = block_size


    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.layers(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #just get last <block_size> tokens
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            #get logits for last time step only
            logits = logits[:, -1, :] # (B,vocab_size)
            probs = F.softmax(logits, dim=-1) #(B,vocab_size)
            #sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append to sequence
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx