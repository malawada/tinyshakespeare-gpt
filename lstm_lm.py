import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1)

class LSTMLanguageModel(nn.Module):
    '''simple LSTM language model that predicts the next character.'''

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.lstm = nn.LSTM(vocab_size, 128, batch_first=True, num_layers=2)
        self.linear = nn.Linear(128, vocab_size)
    
    def forward(self, idx, targets=None):
        '''
        idx: (B,T) tensor of ints
        targets: (B,T) tensor of ints
        '''
        x = self.token_embedding_table(idx)
        x, _ = self.lstm(x)
        logits = self.linear(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        idx: (B,T) array of indices in the current context
        '''
        for _ in range(max_new_tokens):
            logits, _ = self(idx) #get predictions
            logits = logits[:, -1, :] #focus on last timestep only. result is (B,C) tensor.
            probs = F.softmax(logits, dim=-1) #get probabilities for each class in vocabulary
            idx_next = torch.multinomial(probs, num_samples=1) #sample from distribution. result (B,1) tensor.
            idx = torch.cat([idx, idx_next], dim=-1) #append to context
        return idx