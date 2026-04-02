import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Linear projection from d_model → vocab_size.

    Returns RAW LOGITS (no softmax).
    CrossEntropyLoss in train.py already applies log-softmax internally —
    applying it here would double it and cause NaN loss.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)   # raw logits → (batch, seq_len, vocab_size)
