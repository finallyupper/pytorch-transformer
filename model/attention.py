import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot Product Attention Module.

    Args:
        dropout_p (float): Dropout probability. Default is 0.1.

    Inputs:
        Q (torch.Tensor): Queries of shape (batch_size, n_heads, tgt_seq_len, d_k).
        K (torch.Tensor): Keys of shape (batch_size, n_heads, src_seq_len, d_k).
        V (torch.Tensor): Values of shape (batch_size, n_heads, src_seq_len, d_v).
        mask (torch.Tensor or None): Mask tensor of shape (batch_size, 1, tgt_seq_len, src_seq_len).
            Masked positions should be filled with 0, unmasked with 1.

    Outputs:
        torch.Tensor: Output tensor of shape (batch_size, n_heads, tgt_seq_len, d_v).
    """
    def __init__(self, dropout_p=0.1) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_p = dropout_p

    def forward(self, Q, K, V, mask) -> torch.Tensor:
        """
        Compute the scaled dot-product attention.
            Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V   (1)

        Args:
            Q (torch.Tensor): Queries of shape (batch_size, n_heads, tgt_seq_len, d_k).
            K (torch.Tensor): Keys of shape (batch_size, n_heads, src_seq_len, d_k).
            V (torch.Tensor): Values of shape (batch_size, n_heads, src_seq_len, d_v).
            mask (torch.Tensor or None): Mask tensor of shape (batch_size, 1, tgt_seq_len, src_seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_heads, tgt_seq_len, d_v).
        """
        d_k = Q.size(-1)  # Q shape : batch_size x n_heads x tgt_seq_len x d_k
        scale_factor = 1 / math.sqrt(d_k)

        attn_scores = Q @ K.transpose(-2, -1) * scale_factor # batch_size x n_heads x tgt_seq_len x src_seq_len

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        
        attn_weight = torch.softmax(attn_scores, dim=-1)

        if self.dropout_p > .0:
            attn_weight = torch.dropout(attn_weight, p=self.dropout_p, train=True)
        
        outputs = attn_weight @ V 

        return outputs # batch_size x n_heads x tgt_seq_len x d_k
    

class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q, K, V) = Concat(head1, ..., headh)W
        where headi = Attention(QW', KW'', VW''')
    
        Q : batch x tgt_seq_len x d_model   W' : d_model x d_k
        K : batch x src_seq_len x d_model   W'' : d_model x d_k
        V : batch x src_seq_len x d_model   W''' : d_model x d_v
        d_k = d_v = d_model / h = (64)
        
    Args:
        Q (torch.Tensor): Queries of shape (batch_size, tgt_seq_len, d_model)
        K (torch.Tensor): Keys of shape (batch_size, src_seq_len, d_model)
        V (torch.Tensor): Values of shape (batch_size, src_seq_len, d_model)
        mask (torch.Tensor or None): Mask tensor of shape (batch_size, 1, tgt_seq_len, src_seq_len)

    Shapes:
        Input:
            - Q: (batch_size, tgt_seq_len, d_model)
            - K: (batch_size, src_seq_len, d_model)
            - V: (batch_size, src_seq_len, d_model)
            - mask: (batch_size, 1, tgt_seq_len, src_seq_len) or None
        Output:
            - output: (batch_size, tgt_seq_len, d_model)
    
    """
    def __init__(self, d_model=512, n_heads=8, dropout_p=0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model 
        self.n_heads = n_heads 
        self.d_head = d_model // n_heads # split

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout_p=dropout_p)

    def split_head(self, emb) -> torch.Tensor:
        # batch_size x seq_len x d_model  -> batch_size, num_heads, seq_len, d_head
        batch_size, seq_len, d_model = emb.size()
        return emb.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    
    def forward(self, Q, K, V, mask = None) -> torch.Tensor:
        Q, K, V = self.wq(Q), self.wk(K), self.wv(V) 

        Q = self.split_head(Q) # batch_size x n_heads x seq_len x d_head
        K = self.split_head(K)
        V = self.split_head(V)

        # Apply Attention
        attn_heads = self.attention(Q, K, V, mask=mask)

        # Concat 
        batch_size, _, seq_len, d_head = attn_heads.size() # batch_size x n_heads x seq_len x d_head
        d_model = d_head * self.n_heads # 64 * 8 = 512
        concated_attn_heads = attn_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model) # batch_size x seq_len x d_model
        output = self.wo(concated_attn_heads)

        return output # batch_size x seq_len x d_model