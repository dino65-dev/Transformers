import torch.nn as nn
from .rms_norm import RMSNorm


class Decoder(nn.Module):
    """Decoder-only stack: N DecoderBlocks + final RMSNorm."""

    def __init__(self, layers: nn.ModuleList, d_model: int = 768) -> None:
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(dim=d_model)

    def forward(self, x, tgt_mask=None, layer_caches=None, use_cache=False):
        """
        Args:
            x           : (batch, seq_len, d_model) — target embeddings
            tgt_mask    : optional causal attention mask
            layer_caches: list of per-layer KV caches (or None)
            use_cache   : whether to return updated KV caches

        Returns:
            x            : (batch, seq_len, d_model) normalised output
            new_layer_caches: list of updated KV caches
        """
        new_layer_caches = []

        for i, layer in enumerate(self.layers):
            cache = None if layer_caches is None else layer_caches[i]
            x, new_cache = layer(x, tgt_mask=tgt_mask, cache=cache, use_cache=use_cache)
            new_layer_caches.append(new_cache)

        return self.norm(x), new_layer_caches
