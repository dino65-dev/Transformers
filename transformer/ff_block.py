

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model : int, d_ff : int, dropout : float) -> None:
        super().__init__()
        self.activation = nn.GELU()
        # First layer tranformation
        self.linear1 = nn.Linear(d_model,d_ff) # w1 & b1
        self.dropout = nn.Dropout(dropout) # prevent overfitting

        #Sceond layer transformation
        self.linear2 = nn.Linear(d_ff, d_model) # w2 & b2

    def forward(self,x):
        # d_model --> dff --> d_model
        return self.linear2(self.dropout(self.activation(self.linear1(x))))