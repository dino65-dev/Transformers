from .input_embeddings import InputEmbeddings
from .gqa import GroupedQueryAttention
from .ff_block import FeedForwardBlock
from .decoder import Decoder
from .decoder_block import DecoderBlock
from .projection_layer import ProjectionLayer
import torch.nn as nn

# Minimal Transformer matching the notebook's interface/state_dict
class Transformer(nn.Module):
    def __init__(self,
                 encoder=None,
                 decoder=None,
                 src_embed=None,
                 tgt_embed=None,
                 src_pos=None,
                 tgt_pos=None,
                 projection_layer=None,
                 use_rope: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.use_rope = use_rope

        if self.decoder is None:
            raise ValueError("Decoder must be provided.")
        if self.tgt_embed is None:
            raise ValueError("Target embeddings (tgt_embed) must be provided.")
        if self.projection_layer is None:
            raise ValueError("Projection layer must be provided.")

    def decode(self, tgt, tgt_mask, layer_caches=None):
        tgt = self.tgt_embed(tgt)
        if self.tgt_pos is not None:
            tgt = self.tgt_pos(tgt)
        output, new_caches = self.decoder(tgt, tgt_mask, layer_caches)
        return output, new_caches

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                           src_seq_len: int, tgt_seq_len: int,
                           d_model: int = 512, N: int = 6, h: int = 8,
                           kv_h: int = 4, dropout: float = 0.1, d_ff: int = 2048):
    # Create embedding layers
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = GroupedQueryAttention(d_model, h, kv_h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(None, decoder, None, tgt_embed, None, None, projection_layer)

    # Initialize params
    for name, p in transformer.named_parameters():
        if p.dim() > 1:
            if 'embedding' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.xavier_uniform_(p, gain=1.0)
        else:
            nn.init.zeros_(p)

    return transformer