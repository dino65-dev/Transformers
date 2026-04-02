import torch.nn as nn

from .input_embeddings import InputEmbeddings
from .gqa import GroupedQueryAttention
from .ff_block import FeedForwardBlock
from .decoder import Decoder
from .decoder_block import DecoderBlock
from .projection_layer import ProjectionLayer


class Transformer(nn.Module):
    """Decoder-only Transformer."""

    def __init__(self, decoder, tgt_embed, projection_layer) -> None:
        super().__init__()
        if decoder is None:
            raise ValueError("Decoder must be provided.")
        if tgt_embed is None:
            raise ValueError("Target embeddings (tgt_embed) must be provided.")
        if projection_layer is None:
            raise ValueError("Projection layer must be provided.")

        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer

    def decode(self, tgt, tgt_mask=None, layer_caches=None, use_cache=False):
        tgt = self.tgt_embed(tgt)
        output, new_caches = self.decoder(
            tgt, tgt_mask=tgt_mask, layer_caches=layer_caches, use_cache=use_cache
        )
        return output, new_caches

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 768,
    N: int = 12,
    h: int = 12,
    kv_h: int = 4,
    dropout: float = 0.1,
    d_ff: int = 3072,
    use_repo: bool = True,
    use_flash: bool = True,
):
    """
    Build a decoder-only Transformer with:
      - Grouped Query Attention (GQA)
      - REPO-Attention: learned continuous positions (RePoModule)
      - Flash-Attention: F.scaled_dot_product_attention

    Args:
        use_repo : enable REPO-Attention learned positions
        use_flash: enable Flash-Attention (requires PyTorch >= 2.0)
    """
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    decoder_blocks = []
    for _ in range(N):
        attention = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=h,
            num_kv_heads=kv_h,
            dropout=dropout,
            use_repo=use_repo,
            use_flash=use_flash,
        )
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        block = DecoderBlock(attention, ff_block, dropout, d_model=d_model)
        decoder_blocks.append(block)

    decoder = Decoder(nn.ModuleList(decoder_blocks), d_model=d_model)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(decoder, tgt_embed, projection_layer)

    # Weight initialisation
    for name, p in transformer.named_parameters():
        if p.dim() > 1:
            if 'embedding' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'repo' in name or 'W_gate' in name or 'W_content' in name or 'W_proj' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)   # RePoModule already does this, harmless
            else:
                nn.init.xavier_uniform_(p, gain=1.0)
        else:
            nn.init.zeros_(p)

    return transformer