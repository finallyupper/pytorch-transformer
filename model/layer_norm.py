import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    LayerNorm(x + Sublayer(x))
    
    mu = 1/H sigma a_i
    std = sqrt(1/H sigma (a_i - mu)^2) 

    Args:
        d_model (int): Dimension of the model.
        eps (float): A small value to avoid division by zero (default=1e-12).
    """
    def __init__(self, d_model, eps=1e-12) -> None:
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x) -> torch.Tensor:
        # x shape: batch_size x seq_len x d_model
        mean = x.mean(-1, keepdim=True) # batch_size x seq_len x 1
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta # batch_size x seq_len x d_model
    