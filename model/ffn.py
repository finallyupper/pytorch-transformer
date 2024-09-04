import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network used in the Transformer model.
    
    This layer consists of two linear transformations with a ReLU activation in between, 
    followed by a dropout layer for regularization.

    Args:
        d_model (int): Dimension of the model (default=512).
        d_ff (int): Dimension of the feed forward network (default=2048).
        dropout_p (float): Dropout probability (default=0.1).
    """
    def __init__(self, d_model= 512, d_ff=2048, dropout_p=0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff) 
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x) -> torch.Tensor: 
        # x shape : batch_size x seq_len x d_model
        out = self.w1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.w2(out)
        return out# batch_size x seq_len x d_model

