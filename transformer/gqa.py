"""
Grouped Query Attention with:
  - REPO-Attention  : learned continuous positions via RePoModule
  - Flash-Attention : torch.nn.functional.scaled_dot_product_attention (PyTorch ≥ 2.0)

Drop-in replacement for the original gqa.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryPositionEmbedding
from .repo_module import RePoModule


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with REPO-Attention + Flash-Attention.

    Args:
        d_model        : embedding dimension
        num_query_heads: number of query heads
        num_kv_heads   : number of key/value heads (must divide num_query_heads)
        dropout        : dropout probability
        bias           : linear projection bias
        use_repo       : use RePo learned positions (True) or standard RoPE (False)
        use_flash      : use Flash-Attention via scaled_dot_product_attention
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        use_repo: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__()

        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_q_head = num_query_heads
        self.num_kv_head = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads
        self.use_flash = use_flash
        self.use_repo = use_repo
        self.dropout_p = dropout

        # Linear projections (no bias by default — matches LLaMA style)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.num_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.num_kv_head * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout (used in non-flash path only; flash handles it internally)
        self.dropout = nn.Dropout(dropout)

        # ---- Position encoding ----
        # RoPE over the full head_dim
        self.rope = RotaryPositionEmbedding(dim=self.head_dim)

        # REPO-Attention position learning module
        if use_repo:
            self.repo = RePoModule(hidden_dim=d_model)
        else:
            self.repo = None

        # Scale for manual attention (flash handles this internally)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        query,
        key=None,
        value=None,
        attn_mask=None,
        is_causal: bool = False,
        need_weights: bool = False,
        cache=None,
    ):
        """
        Args:
            query      : (batch, seq_len, d_model)  — the main input
            key/value  : same shape as query (defaults to self-attention)
            attn_mask  : optional bool/float mask
            is_causal  : if True, apply causal mask (auto-handled by flash)
            need_weights: return attention weights (forces manual path)
            cache      : (past_k, past_v) for KV caching
        """
        # Save the raw hidden states for REPO position learning
        hidden_states = query   # (batch, seq_len, d_model)

        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len, _ = query.shape
        kv_seq_len = key.shape[1]

        # ── 1. Linear projections ──────────────────────────────────────
        q = self.q_proj(query)   # (batch, seq_len, d_model)
        k = self.k_proj(key)     # (batch, kv_seq_len, num_kv * head_dim)
        v = self.v_proj(value)   # (batch, kv_seq_len, num_kv * head_dim)

        # ── 2. Reshape → (batch, seq, heads, head_dim) ────────────────
        q = q.view(batch_size, seq_len,    self.num_q_head,  self.head_dim)
        k = k.view(batch_size, kv_seq_len, self.num_kv_head, self.head_dim)
        v = v.view(batch_size, kv_seq_len, self.num_kv_head, self.head_dim)

        # ── 3. Position encoding (REPO or standard RoPE) ───────────────
        if self.use_repo and self.repo is not None:
            # Learn continuous positions from the full hidden state
            learned_positions = self.repo(hidden_states)   # (batch, seq_len)
            q, k = self.rope.apply_rotary_pos_emb(q, k, positions=learned_positions)
        else:
            # Standard RoPE with integer positions
            q, k = self.rope.apply_rotary_pos_emb(q, k, positions=None)

        # ── 4. Transpose → (batch, heads, seq, head_dim) for attention ─
        q = q.transpose(1, 2)   # (batch, num_q,  seq_len,    head_dim)
        k = k.transpose(1, 2)   # (batch, num_kv, kv_seq_len, head_dim)
        v = v.transpose(1, 2)   # (batch, num_kv, kv_seq_len, head_dim)

        # ── 5. Expand KV heads to match query heads (GQA) ─────────────
        k_expanded = k.repeat_interleave(self.group_size, dim=1)
        v_expanded = v.repeat_interleave(self.group_size, dim=1)

        # ── 6. KV Cache ───────────────────────────────────────────────
        if cache is not None:
            past_k, past_v = cache
            k_expanded = torch.cat((past_k, k_expanded), dim=2)
            v_expanded = torch.cat((past_v, v_expanded), dim=2)
        present_kv = (k_expanded, v_expanded)

        # ── 7. Attention ──────────────────────────────────────────────
        use_flash_here = self.use_flash and not need_weights and attn_mask is None

        if use_flash_here:
            # Flash-Attention via scaled_dot_product_attention (PyTorch ≥ 2.0)
            # Handles causal masking, scaling, and dropout internally
            dropout_p = self.dropout_p if self.training else 0.0
            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )  # (batch, num_q_heads, seq_len, head_dim)

        else:
            # Manual attention path (used when need_weights=True or attn_mask provided)
            attn_scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale

            if is_causal:
                kv_len = k_expanded.shape[2]
                causal_mask = torch.tril(
                    torch.ones(seq_len, kv_len, device=q.device, dtype=torch.bool)
                )
                attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                attn_scores = attn_scores.masked_fill(
                    (1 - attn_mask).bool(), float('-inf')
                )

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v_expanded)

        # ── 8. Merge heads → (batch, seq_len, d_model) ────────────────
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # ── 9. Output projection ───────────────────────────────────────
        output = self.out_proj(attn_output)

        if need_weights and not use_flash_here:
            # Average weights across heads for visualization
            attn_weights = attn_probs.mean(dim=1)   # (batch, seq_len, kv_seq_len)
            return output, attn_weights, present_kv
        else:
            return output, present_kv
