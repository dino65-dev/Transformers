import torch.nn as nn
from rms_norm import RMSNorm

class Decoder(nn.Module):
    # A Decoder can have sevarel decoder blocks
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        #storing the 'DecoderBlock's
        self.layers = layers
        self.norm = RMSNorm() # to normalize the output

    def forward(self, x, encoder_output_key, encoder_output_value, src_mask, tgt_mask, layer_caches=None):
        new_layer_caches = []
        # per layer kv cache
        for i, layer in enumerate(self.layers):
            layer_cache = None if layer_caches is None else layer_caches[i]
            x, new_cache = layer(x, encoder_output_key, encoder_output_value, src_mask, tgt_mask,layer_cache)
            new_layer_caches.append(new_cache)

        return self.norm(x), new_layer_caches
