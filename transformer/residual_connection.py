from .rms_norm import RMSNorm
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm()

    def forward(self, x, sublayer):
        # Ensure x is a tensor
        if isinstance(x, tuple):
            x = x[0]

        # Apply normalization
        normed_x = self.norm(x)

        # Apply sublayer
        sublayer_output = sublayer(normed_x)

        # Handle both cached and non-cached sublayer outputs
        if isinstance(sublayer_output, tuple):
            # Sublayer returned (output, cache)
            output_tensor, cache = sublayer_output
            residual_output = x + self.dropout(output_tensor)
            return residual_output, cache  # Return tuple
        else:
            # Sublayer returned only output tensor
            output_tensor = sublayer_output
            residual_output = x + self.dropout(output_tensor)
            return residual_output  # Return tensor only

