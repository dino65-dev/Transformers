import torch.nn as nn
from .residual_connection import ResidualConnection


class DecoderBlock(nn.Module):
    """
    Single decoder layer: self-attention + feed-forward with residual connections.
    Supports both training (no cache) and inference (KV cache) modes.
    """

    def __init__(self, masked_attention_block, feed_forward_block, dropout, d_model: int = 768):
        super().__init__()
        self.masked_attention = masked_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout, d_model=d_model) for _ in range(2)]
        )

    def forward(self, x, tgt_mask=None, cache=None, use_cache=False):
        """
        Args:
            x        : (batch, seq_len, d_model)
            tgt_mask : optional attention mask
            cache    : (past_k, past_v) for KV caching during inference
            use_cache: whether to pass cache into attention

        Returns:
            x         : (batch, seq_len, d_model)
            present_kv: (k, v) tensors — always returned, None during training
        """
        # ── Self-Attention ────────────────────────────────────────────
        # We pass `x` as hidden_states so the RePoModule inside GQA can
        # read the full embedding for position learning.
        def self_attn_fn(normed_x):
            return self.masked_attention(
                query=normed_x,
                key=normed_x,
                value=normed_x,
                attn_mask=tgt_mask,
                is_causal=True,      # causal mask handled by Flash-Attention
                cache=cache if use_cache else None,
            )

        result = self.residual_connection[0](x, self_attn_fn)

        # ResidualConnection returns (tensor, cache) because GQA returns a tuple
        if isinstance(result, tuple):
            x, present_kv = result
        else:
            x, present_kv = result, None

        # ── Feed-Forward ──────────────────────────────────────────────
        x = self.residual_connection[1](x, self.feed_forward)

        return x, present_kv
