class ProjectionLayer(nn.Module):
    # projection layer is the output of ffn from decoder and the applied on liner,softmax layer
    def __init__(self, d_model : int , vacab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vacab_size) # Linear layer

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)  # Applying the log Softmax function to the output
