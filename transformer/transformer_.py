from typing import Optional
from input_embeddings import InputEmbeddings
from positional_encoding import PositionalEncoding
from projection_layer import ProjectionLayer
from decoder import Decoder
import torch.nn as nn
class Transformer(nn.Module):

    """This takes in the encoder and decoder, as well the embeddings for the source
     and target language. It also takes in the postional encoding for the source and target language,
      as well as projection layer """

    def __init__(self,
                 decoder: Optional[Decoder] = None,
                 src_embed: Optional[InputEmbeddings] = None,
                 tgt_embed: Optional[InputEmbeddings] = None,
                 src_pos: Optional[PositionalEncoding] = None,
                 tgt_pos: Optional[PositionalEncoding] = None,
                 projection_layer: Optional[ProjectionLayer] = None,
                use_rope: bool = True) -> None:
        super().__init__()
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.use_rope = use_rope

    # Validate configuration
        if self.use_rope and (self.src_pos is not None or self.tgt_pos is not None):
            print("Warning: Using RoPE with separate positional encodings. "
                  "Consider setting src_pos=None, tgt_pos=None for pure RoPE.")


    #Decoder
    # tgt_embed --> right shifted embedding
    #src_embed --> tensor of embed
    #src_pos --> normal pos of embbedings(encoder)
    #tgt_pos --> same formula but for decoder input
    def decode(self, encoder_output_key, encoder_output_value, src_mask, tgt , tgt_mask, layer_caches = None):
        if self.tgt_embed is None or self.decoder is None:
            raise ValueError("Decoder components are not properly initialized.")
        tgt = self.tgt_embed(tgt) # Applying target embeddings to the input target language (tgt)
        # Apply positional encoding only if not using RoPE or explicitly provided
        if not self.use_rope and self.tgt_pos is not None:
            tgt = self.tgt_pos(tgt) # Applying target positional encoding to the target embeddings
        elif self.use_rope and self.tgt_pos is not None:
            # Optional: Apply traditional pos encoding alongside RoPE
            tgt = self.tgt_pos(tgt) # Applying target positional encoding to the target embeddings

        output, new_caches = self.decoder(tgt, encoder_output_key, encoder_output_value, src_mask, tgt_mask, layer_caches)
        # Returing the target embeddings, the output of encoder, and both source and target masks
        # The target mask ensures that the model won't see future elements of the sequence
        return output , new_caches

    #Applying projection layer with the Softmax Function to the decoder output
    def project(self, x):
        if self.projection_layer is None:
            raise ValueError("Projection layer is not initialized.")
        return self.projection_layer(x)