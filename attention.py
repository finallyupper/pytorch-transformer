import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 

"""

nn.Linear(in_features, out_features)
- y = xW^T + b
- W, b = 자동 초기화 (random) -> 이후 backpropagation에 의해 업데이트된다.

nn.bmm을 통해 batch matrix multiplication 사용하는 것도 가능하다. 
"""

class SelfAttention(nn.Module):
    def __init__(self, d_model): # d_model == d_k
        super(SelfAttention, self).__init__()
        self.d_model = d_model 
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        d_k = query.size(-1)
        score = query @ key.transpose(-2, -1) / torch.sqrt(d_k, dtype=torch.float32)
        attn_weight = torch.softmax(score, dim=-1) # softmax는 주로 torch.softmax를 통해 직접 적용하는 것이 일반적
        output = attn_weight @ value 
        return output 

# reference : https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
def ScaledDotProductAttention(query, key, value, attn_mask=None, dropout_p=.0, scale=None)->torch.Tensor:
    """
    Scaled Dot Produc Attention
    args:
        Q : batch x target seq len x d_k
        K : batch x source seq len x d_k
        v : batch x source seq len x d_v
    """
    assert query.size(-1) == key.size(-1), "the embedding dimension of Q, K should be same."
    d_k = query.size(-1)
    L, S = query.size(-2), key.size(-2)

    if scale is not None:
        scale_factor = 1 / math.sqrt(d_k)
    else:
        scale_factor = 1

    # Matrix Multiplication
    attn_scores = query @ key.transpose(-2, -1)
    # scale
    attn_scores /= scale_factor 
     
    # Mask (Opt.)
    if attn_mask is not None:
        attn_mask = torch.tril(torch.ones((L, S)), diagonal=0)
        attn_mask.masked_fill_(torch.logical_not(attn_mask), float('-inf'))
        attn_align = attn_scores + attn_mask 
    else:
        attn_align = attn_scores
    # Softmax
    attn_weight = torch.softmax(attn_align, dim=-1)
    attn_weight = torch.dropout(attn_mask, p=dropout_p, train=True)
    # Matrix Multiplication
    output = attn_weight @ value 
    return output 


def ScaledDotProductAttention(query, key, value, attn_mask=None, dropout_p=.0, scale=None)->torch.Tensor:
    """
    Scaled Dot Product Attention
    args:
        Q : batch x (head) x target seq len x d_k
        K : batch x (head) x source seq len x d_k
        v : batch x (head) x source seq len x d_v
    """
    assert query.size(-1) == key.size(-1), "the embedding dimension of Q, K should be same."
    d_k = query.size(-1)
    L, S = query.size(-2), key.size(-2)

    if scale is not None:
        scale_factor = 1 / math.sqrt(d_k)
    else:
        scale_factor = 1

    # Matrix Multiplication
    # 마지막 두개의 차원에서 연산이 이루어짐.
    attn_scores = query @ key.transpose(-2, -1)
    # scale
    attn_scores /= scale_factor 
     
    # Mask (Opt.)
    if attn_mask is not None:
        attn_mask = torch.tril(torch.ones((L, S)), diagonal=0)
        attn_mask.masked_fill_(torch.logical_not(attn_mask), float('-inf'))
        attn_align = attn_scores + attn_mask 
    else:
        attn_align = attn_scores
    # Softmax
    attn_weight = torch.softmax(attn_align, dim=-1)
    attn_weight = torch.dropout(attn_mask, p=dropout_p, train=True)
    # Matrix Multiplication
    output = attn_weight @ value 
    return output 


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_head):
        self.d_model = d_model 
        self.d_head = d_head 
        self.attention = ScaledDotProductAttention()
        self.q = nn.Linear(d_model, d_head)
        self.k = nn.Linear(d_model, d_head)
        self.v = nn.Linear(d_model, d_head)

    def forward(self, x):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        # tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # # it is similar with group convolution (split by number of heads)

        return self.attention(query, key, value, scale=True)
    
# 헤드를 여러개 두어서 다양한 측면의 정보를 받아들일 수 있게 하자.
# reference : https://seungseop.tistory.com/29 
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    d_k = d_v = d_model / h
    """
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model 
        self.d_head = d_model // n_heads # split 
        self.attn_heads = nn.ModuleList(
            [AttentionHead(self.d_model, self.d_head) for _ in range(n_heads)]
        )
        self.linear = nn.Linear(d_model, d_model) 
    def forward(self, x):
        attn_heads = [head(x) for head in self.attn_heads]
        concated_attn_heads = torch.cat(attn_heads, dim=-1) # batch x ... x d_k(= d_head * h)
        output = self.linear(concated_attn_heads)
        return output 


class PositionalEncoding(nn.Moudle):
    """
    Positional Encoding
    self-attention은 permutation equivalent하기 때문에 input들의 order를 고려하지 못함.
    -> Order of the sequence 정보를 부여하자. (inject some information)

    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))     -> i mod 2 == 0
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
    
    where pos: position, i: dimension
    
    - max_len determines how far the position can have an effect on a token
    - 논문의experiment에서 PE의 dropout 값의 디폴트를 0.1로함.
    torch.arange(0, d_model, 2) : 2i를 표현

    """
    def __init__(self, d_model, dropout_p = 0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_p)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 2-dim 
        division = torch.exp(torch.arange(0, d_model, 2).float() / d_model * -math.log(10000.0))
        pe[:, ::2] = torch.sin(pos * division)
        pe[:, 1::2] = torch.cos(pos * division)

        self.positional_encoding = pe
    def forward(self, x):
        x = x.transpose(0, -2) # batch size x sequence length x d_model  
        seq_len = x.size(0) # sequence length x batch size x d_model
        return self.dropout(x + self.positional_encoding[:seq_len, :])
    

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model=512, d_ff=2048, dropout_p = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model # dim of input/output
        self.d_ff = d_ff # dim of inner layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout_p)
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        return self.linear2(out)
    

"""
두개의 sub layers(Multi-head attention, position-wise fully connected feed-forward network)
는 모두 layer normalization이 뒤에 붙어 있음
"""
class LayerNormalization(nn.Module):
    """
    layer Normalization은 batch Normalization이 가지고 있던 Batch에 대한 의존도를 제거하고 
    batch가 아닌 layer를 기반으로하여 Normalization을 수행

    LayerNorm(x + Sublayer(x))

    H : # of hidden units in layer
    단일 layer에 있는 모든 hidden unit들은 동일한 mu, std를 공유

    mu = 1/H sigma a_i
    std = sqrt(1/H sigma (a_i - mu)^2) 

    """
    def __init__(self, d_model):
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
    def forward(self, x):
        x = x # shape: [batch, seq_len, d_model]