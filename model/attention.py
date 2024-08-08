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



def ScaledDotProductAttention(query, key, value, attn_mask=None, dropout_p=.0, scale=None)->torch.Tensor:
    """
    Scaled Dot Product Attention
    args:
        Q : batch x (num_heads) x target seq len x d_k
        K : batch x (num_heads) x source seq len x d_k
        v : batch x (num_heads) x source seq len x d_v
    """
    assert query.size(-1) == key.size(-1), "the embedding dimension of Q, K should be same."
    d_k = query.size(-1)
    L, S = query.size(-2), key.size(-2)

    scale_factor = 1 / math.sqrt(d_k) if scale is not None else 1.0

    # Matrix Multiplication & scale
    # 마지막 두개의 차원에서 연산이 이루어짐.
    attn_scores = query @ key.transpose(-2, -1) * scale_factor # [batch_size, num_heads, tgt_seq_len, source_seq_len]
     
    # Mask (Opt.)
    if attn_mask is not None: # [batch_size, 1, tgt_seq_len, tgt_seq_len]
        # attn_mask = torch.tril(torch.ones((L, S)), diagonal=0) # target seq_len x source seq_len
        # attn_mask.masked_fill_(torch.logical_not(attn_mask), float('-inf'))
        # attn_align = attn_scores + attn_mask 
        attn_scores = attn_scores.masked_fill(attn_mask==0, -1e9) #Modified: reference https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

    # Softmax
    attn_weight = torch.softmax(attn_scores, dim=-1)

    if dropout_p > .0:
        attn_weight = F.dropout(attn_weight, p=dropout_p, train=True) # [batch, d_head, seq_len, seq_len]
    # Matrix Multiplication
    output = attn_weight @ value
    return output # batch x num_head x seq_len x d_head [64, 8, 100, 64]


class AttentionHead(nn.Module):
    """
    q, k, v shape : [batch_size, num_heads, seq_len, d_head]
    """
    def __init__(self, d_model, d_head):
        super(AttentionHead, self).__init__()
        self.d_model = d_model 
        self.d_head = d_head # 64
#        self.attention = ScaledDotProductAttention()
        # self.q = nn.Linear(d_model, d_head)
        # self.k = nn.Linear(d_model, d_head)
        # self.v = nn.Linear(d_model, d_head)

    def forward(self, q, k, v, attn_mask=None):
        # query = self.q(q) # [batch, seq_len, d_head]
        # key = self.k(k)
        # value = self.v(v)

        # tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # # it is similar with group convolution (split by number of heads)

        return ScaledDotProductAttention(q, k, v, scale=True, attn_mask=attn_mask)

# 헤드를 여러개 두어서 다양한 측면의 정보를 받아들일 수 있게 하자.
# reference : https://seungseop.tistory.com/29 
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    

    MultiHead(Q, K, V) = Concat(head1, ..., headh)W
        where headi = Attention(QW', KW'', VW''')
        
        Q : batch x tgt_seq_len x d_model
        K : batch x src_seq_len x d_model
        V : batch x src_seq_len x d_model

        W' : d_model x d_k
        W'' : d_model x d_k
        W''' : d_model x d_v
        d_k = d_v = d_model / h = (64)

    input shape: batch x ... x d_model
    output shape: batch x ... x d_model     

    """
    def __init__(self, d_model=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model 
        self.n_heads = n_heads
        self.d_head = d_model // n_heads # split
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.attn_head = AttentionHead(self.d_model, self.d_head)
        self.attn_heads = nn.ModuleList(
            [AttentionHead(self.d_model, self.d_head) for _ in range(n_heads)]
        )
        self.linear = nn.Linear(d_model, d_model) # d_model = d_head * h

    def split_head(self, emb):
        # [ batch_size, seq_len, d_model ] -> [batch_size, num_heads, seq_len, d_head]
        batch_size, seq_len, d_model = emb.size()
        return emb.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, q, k, v, attn_mask=None):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)

        # attn_heads = [head(q, k, v, attn_mask) for head in self.attn_heads] # each [batch_size, num_heads, seq_len, d_head] ->
        # concated_attn_heads = torch.cat(attn_heads, dim=-1) # batch x num_heads x seq_len x d_model
        
        attn_heads = self.attn_head(q, k, v, attn_mask) # [batch_size, num_heads, seq_len, d_head]
        batch_size, _, seq_len, d_head = attn_heads.size()
        d_model = d_head * self.n_heads # 64 * 8 = 512
        concated_attn_heads = attn_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model) #NOTE: contiguous함수를 사용할 경우 강제로 메모리를 새로 할당해 연속적으로 만든 후 view를 사용하면 그 모양대로 수정 가능하다.
        output = self.linear(concated_attn_heads)
        return output # batch x seq_len x d_model 


class PositionalEncoding(nn.Module):
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

        self.pe = torch.zeros(max_len, d_model)
        self.pe.requires_grad = False 
        
        pos = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 2-dim 

        division = torch.arange(0, d_model, step=2).float() 
        # division = torch.exp(torch.arange(0, d_model, step=2).float() / d_model * -math.log(10000.0))
        self.pe[:, ::2] = torch.sin(pos * (10000 ** (division / d_model)))
        self.pe[:, 1::2] = torch.cos(pos * (10000 ** (division / d_model)))

    def forward(self, x):
        # x = x.transpose(0, -2) # batch size x sequence length x d_model  -> seq_len x batch size x d_model [100x64x512]
        seq_len = x.size(-2) # sequence length x batch size x d_model
        return self.dropout(x + self.pe[:seq_len, :]) # 2nd term shape: 100x512= seq_len x d_model
    

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
        out = self.dropout(out) # -> [batch_size, seq_len, d_ff] = [64, 100, 2048]
        return self.linear2(out) # -> [64, 100, 512]
    

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
    def __init__(self, d_model, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x shape: batch_size x seq_len x d_model
        mean = x.mean(-1, keepdim=True) # batch_size x seq_len x 1
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
    

# # reference : https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
# def ScaledDotProductAttention(query, key, value, attn_mask=None, dropout_p=.0, scale=None)->torch.Tensor:
#     """
#     Scaled Dot Produc Attention
#     args:
#         Q : batch x target seq len x d_k
#         K : batch x source seq len x d_k
#         v : batch x source seq len x d_v
#     """
#     assert query.size(-1) == key.size(-1), "the embedding dimension of Q, K should be same."
#     d_k = query.size(-1)
#     L, S = query.size(-2), key.size(-2)

#     if scale is not None:
#         scale_factor = 1 / math.sqrt(d_k)
#     else:
#         scale_factor = 1

#     # Matrix Multiplication
#     attn_scores = query @ key.transpose(-2, -1)
#     # scale
#     attn_scores /= scale_factor 
     
#     # Mask (Opt.)
#     if attn_mask is not None:
#         attn_mask = torch.tril(torch.ones((L, S)), diagonal=0)
#         attn_mask.masked_fill_(torch.logical_not(attn_mask), float('-inf'))
#         attn_align = attn_scores + attn_mask 
#     else:
#         attn_align = attn_scores
#     # Softmax
#     attn_weight = torch.softmax(attn_align, dim=-1)
#     attn_weight = torch.dropout(attn_mask, p=dropout_p, train=True)
#     # Matrix Multiplication
#     output = attn_weight @ value 
#     return output 
