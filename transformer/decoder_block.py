class DecoderBlock(nn.Module):
    def __init__(self, masked_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.masked_attention = masked_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, tgt_mask, cache=None, use_cache=False):
        # Self-attention block
        if use_cache:
            # During inference with caching
            result = self.residual_connection[0](
                x,
                lambda x: self.masked_attention(
                    query=x, key=x, value=x,
                    attn_mask=tgt_mask,
                    is_causal=True,
                    cache=cache
                )
            )
            
            # Handle tuple return from residual connection
            if isinstance(result, tuple):
                x, self_attn_cache = result
            else:
                x, self_attn_cache = result, None
        else:
            # During training without caching
            x = self.residual_connection[0](
                x,
                lambda x: self.masked_attention(
                    query=x, key=x, value=x,
                    attn_mask=tgt_mask,
                    is_causal=True
                )
            )
            self_attn_cache = None
        
        # Feed forward block (no cache)
        x = self.residual_connection[1](x, self.feed_forward)
        
        # Return based on mode
        if use_cache:
            return x, self_attn_cache
        else:
            return x  # Return only tensor during training

     
