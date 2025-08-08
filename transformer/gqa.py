import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from .rope_helper import apply_rotary_pos_emb
from .rope import RotaryPositionalEmbedding
class GroupedQueryAttention(nn.Module):
    """
        Grouped Query Attention

        Args:
            d_model: Embedding dimension
            num_query_heads: Number of query heads
            num_kv_heads: Number of key-value heads (must divide num_query_heads)
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
            rope_percentage: Decides what percentage of embeddings will be used for rope
        """
    def __init__(self, d_model : int , num_query_heads : int, num_kv_heads : int, dropout = 0.1, bias = False, rope_percentage = 0.5) -> None:
        super().__init__()

        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_q_head = num_query_heads
        self.num_kv_head = num_kv_heads
        # per head dim
        self.head_dim = d_model // num_query_heads
        #how many query heads share a single KV head
        self.group_size = num_query_heads // num_kv_heads

        #rope initialization
        self.rope_percentage = rope_percentage
        self.rope_dim = int(self.head_dim * rope_percentage)
        if self.rope_dim > 0:
            self.rotary_pe = RotaryPositionalEmbedding(self.rope_dim)

        #Linear projections
        self.q_proj = nn.Linear(d_model,d_model,bias=bias)
        self.k_proj = nn.Linear(d_model,self.num_kv_head * self.head_dim,bias=bias)
        self.v_proj = nn.Linear(d_model,self.num_kv_head * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model,d_model,bias=bias)
        self.dropout = nn.Dropout(dropout) # prevent overfitting
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self,query, key = None, value = None, attn_mask = None,is_causal = False, need_weigths = False,cache = None):
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size = query.shape[0]
        seq_len = query.shape[1]
        kv_seq_length = key.shape[1]

        #project queries , keys, values
        q = self.q_proj(query) #[batch, seq_len, d_model]
        k = self.k_proj(key) #[batch, kv_seq_len , num_kv_heads * head dim]
        v = self.v_proj(value) # [batch, kv_seq_len, num_kv_heads* head_dim]

        # Reshape and transpose for mha
        q = q.view(batch_size,seq_len,self.num_q_head,self.head_dim).transpose(1,2) # [batch, num_query_heads, seq_len, head_dim]
        k = k.view(batch_size, kv_seq_length,self.num_kv_head,self.head_dim).transpose(1,2) # [batch, num_kv_head, kv_seq_length, head_dim]
        v = v.view(batch_size, kv_seq_length, self.num_kv_head, self.head_dim).transpose(1,2) #[batch, num_kv_head, kv_seq_length, head_dim]

    # ============ If you are going to use PE then don't use RoPE and vice_versa ========================

        # Applying Rope
        if self.rope_dim > 0:
            #split into RoPE and non-RoPE parts
            q_rope, q_pass = q[...,:self.rope_dim], q[...,self.rope_dim:]
            k_rope , k_pass = k[...,:self.rope_dim] , k[...,self.rope_dim:]

            # Apply rotary embeddings to queries
            cos_q , sin_q = self.rotary_pe(q_rope)
            q_rope = apply_rotary_pos_emb(q_rope,cos_q,sin_q)

            #Apply rotary embeddings to keys
            cos_k , sin_k = self.rotary_pe(k_rope)
            k_rope = apply_rotary_pos_emb(k_rope,cos_k,sin_k)

            # concatenate back
            q = torch.cat([q_rope,q_pass],dim=-1)
            k = torch.cat([k_rope, k_pass],dim=-1)

    #=======================================================================================================

        #Expand keys and values to match query heads
        # Each group of query heads shares the same kv heads
        #after learning the learned matrices of key is copied into (k_head * group_size) total
        k_expanded = k.repeat_interleave(self.group_size, dim =1) # [batch, num_query_heads, kv_seq_len, head_dim]
        v_expanded = v.repeat_interleave(self.group_size,dim=1)  # [batch, num_query_heads, kv_seq_len, head_dim]

        # KV caching
        if cache is not None:
            past_key , past_value = cache
            k_expanded = torch.cat((past_key,k_expanded),dim=2)
            v_expanded = torch.cat((past_value,v_expanded),dim=2)
        present_kv = (k_expanded,v_expanded)

        # compute attention scores
        # query : seq_len, head_dim * key: head_dim ,kv_seq_len
        attn_scores = torch.matmul(q,k_expanded.transpose(-2,-1)) * self.scale # [batch, num_query_heads, seq_len, kv_seq_len]

        # Apply masks
        if is_causal:
            causal_mask = torch.tril(torch.ones(seq_len,kv_seq_length,device=q.device,dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask,float('-inf')) # inverse the causal mask and where is true replace that with -infinity

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill((1 - attn_mask).to(bool), float('-inf'))

        # compute attention probabilites
        attn_probs = F.softmax(attn_scores ,dim=-1)
        attn_probs = self.dropout(attn_probs)

        #Apply attention to values
        attn_output = torch.matmul(attn_probs, v_expanded) ## [batch, num_query_heads, seq_len, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1,2).contiguous() # [batch, seq_len, num_query_heads, head_dim]

        attn_output = attn_output.view(batch_size,seq_len,self.d_model) # [batch, seq_len, embed_dim]

        #Final output projection
        output = self.out_proj(attn_output)

        if need_weigths:
            #Average attention weights across heads for visualization
            attn_weights = attn_probs.mean(dim=1) # [batch, seq_len, kv_seq_len]
            return output, attn_weights, present_kv
        else: return output , present_kv
