import torch.nn as nn
import torch
class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, d_model, base = 10000) -> None:
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0,d_model,2).float() / d_model)) # this is out theta(i)
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self,x, seq_dim = 2): # seq_dim=2 for [batch, heads, seq_len, head_dim]
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len,device=x.device).type_as(self.inv_freq) # position_indices[0 --> seq_len-1]
            freqs = torch.einsum('i,j -> ij',t,self.inv_freq) # t ⊗ inv_freq (outer product)
            emb = torch.cat((freqs,freqs),dim= -1).to(x.device) # creates [cos,sin,cos,sin] pattern, more importantly we are repeating cause it doesn't dimenstion mismatch at the broadcasting time
            self.cos_cached = emb.cos()[None,None,:,:] # [1, 1, seq_len, head_dim]
            self.sin_cached = emb.sin()[None,None,:,:]
        return self.cos_cached , self.sin_cached
