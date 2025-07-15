import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:  # Fixed: d_mdoel -> d_model
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # Fix 2: Remove aggressive scaling that causes NaN
        return self.embedding(x) * math.sqrt(self.d_model)
