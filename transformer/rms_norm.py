class RMSNorm(nn.Module):
    def __init__(self,dim: int = 128, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        # The learnable scaling parameter, with a size of the feature dimension
        self.gamma = nn.Parameter(torch.ones(self.dim))

    def _norm(self, x):
        # Calculate the reciprocal of the square root for efficiency
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # ✅ Add input validation
        if isinstance(x, tuple):
            # If input is tuple, use only the first element (the actual tensor)
            x = x[0]
            print("Warning: RMSNorm received tuple input, using first element")
        
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"RMSNorm expected tensor input, got {type(x)}")
        
        # Normalize and then scale
        return self.gamma * self._norm(x.float()).type_as(x)