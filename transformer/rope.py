"""
RotaryPositionEmbedding — upgraded version from REPO-Attention.

Supports:
  - Standard RoPE  : integer positions [0, 1, 2, ...]
  - RePo           : continuous learned positions [0.5, 2.3, 1.1, ...]
"""

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """
    RoPE: Encodes position via rotation matrices in complex plane.

    Key insight: Relative position (j-i) naturally emerges from
    rotation difference between positions j and i.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        """
        Args:
            dim: Per-head embedding dimension (must be even).
            max_seq_len: Maximum sequence length for cache.
            base: Base for frequency calculation.
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # θ_i = base^(-2i/d)  shape: (dim//2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    # ------------------------------------------------------------------
    def _build_cache(self, seq_len: int):
        """Precompute cos/sin for integer positions [0 .. seq_len-1]."""
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)          # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)                # (seq_len, dim)
        # shape: (1, seq_len, 1, dim)  →  broadcast-friendly for (batch, seq, heads, dim)
        self.register_buffer('cos_cached', emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :], persistent=False)
        self.max_seq_len = seq_len

    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """[x1, x2, ...] → [-x2, x1, ...]  (90° rotation in complex plane)"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    # ------------------------------------------------------------------
    def _interpolate_rotary(self, positions: torch.Tensor):
        """
        Compute cos/sin for CONTINUOUS positions (RePo).

        Args:
            positions: (batch, seq_len)  — learned float positions

        Returns:
            cos, sin: (batch, seq_len, 1, dim)
        """
        # (batch, seq_len, 1) * (1, 1, dim//2) → (batch, seq_len, dim//2)
        freqs = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)            # (batch, seq_len, dim)
        cos = emb.cos().unsqueeze(2)                        # (batch, seq_len, 1, dim)
        sin = emb.sin().unsqueeze(2)
        return cos, sin

    # ------------------------------------------------------------------
    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor = None,
    ):
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: (batch, seq_len, num_heads, head_dim)
            k: (batch, seq_len, num_kv_heads, head_dim)
            positions: (batch, seq_len) learned float values, or None for standard RoPE

        Returns:
            q_rot, k_rot with same shapes as input
        """
        seq_len = q.shape[1]

        if positions is None:
            # Standard RoPE — use precomputed cache
            if seq_len > self.max_seq_len:
                self._build_cache(seq_len)
            cos = self.cos_cached[:, :seq_len, :, :]   # (1, seq_len, 1, dim)
            sin = self.sin_cached[:, :seq_len, :, :]
        else:
            # RePo — continuous positions
            cos, sin = self._interpolate_rotary(positions)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
