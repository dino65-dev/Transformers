from .input_embeddings import InputEmbeddings
from .gqa import GroupedQueryAttention
from .ff_block import FeedForwardBlock
from .decoder import Decoder
from .decoder_block import DecoderBlock
from .projection_layer import ProjectionLayer
from .transformer_ import Transformer
import torch.nn as nn
def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                           src_seq_len: int, tgt_seq_len: int,
                           d_model: int = 512, N: int = 6, h: int = 8,
                           kv_h: int = 4, dropout: float = 0.1, d_ff: int = 2048):

    # Create embedding layers
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding
    #tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = GroupedQueryAttention(d_model, h, kv_h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        decoder_block = DecoderBlock(decoder_self_attention_block,
                                   feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
                                                                   #tgt_pos --> uisng RoPE
    transformer = Transformer(None, decoder, None, tgt_embed, None, None, projection_layer)


    for name, p in transformer.named_parameters():
        if p.dim() > 1:
            if 'embedding' in name:
                # Use smaller initialization for embeddings
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.xavier_uniform_(p, gain=1.0)
        else:
            nn.init.zeros_(p)

    return transformer