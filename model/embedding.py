import torch
from torch import nn

class TokenEmbedding(nn.Embedding):
    """Reference = https://github.com/gusdnd852"""
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=2)

class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding function for Transformer models.

    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))     -> i mod 2 == 0
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
    
    where pos: position, i: dimension
    - max_len determines how far the position can have an effect on a token

    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the sequence.
        device (torch.device or None): Device to allocate the positional encodings.

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

    Output:
        torch.Tensor: Positional encodings for the input tensor of shape (seq_len, d_model).
    """
    def __init__(self, d_model = 512, max_len=5000, device=None) -> None:
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model, device=device)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        division = torch.arange(0, d_model, step=2, device=device).float() 
        self.pe[:, ::2] = torch.sin(pos * (10000 ** (division / d_model)))
        self.pe[:, 1::2] = torch.cos(pos * (10000 ** (division / d_model)))
    def forward(self, x) -> torch.Tensor:
        # x shape : batch_size x seq_len
        seq_len = x.size(1) 
        return self.pe[:seq_len, :] # seq_len x d_model

class TransformerEmbedding(nn.Module):
    """
    Combines token embeddings with positional encodings and applies dropout.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the sequence.
        dropout_p (float): Dropout probability.
        device (torch.device or None): Device to allocate the embeddings and positional encodings.

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

    Output:
        torch.Tensor: Combined token embeddings and positional encodings of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, vocab_size, d_model, max_len, dropout_p=0.1, device=None) -> None:
        super(TransformerEmbedding, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, device=device)
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.vocab_size = vocab_size
    def forward(self, x) -> torch.Tensor: # batch_size x src_max_len
        emb_x = self.token_embedding(x) # batch_size x seq_len x d_model
        out = self.dropout(emb_x + self.positional_encoding(x))
        return out # batch_size x seq_len x d_model
    