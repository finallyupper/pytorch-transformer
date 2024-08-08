import torch 
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention, PositionwiseFeedForward, LayerNormalization

class EncoderBlock(nn.Module):
    """
    stack of n=6 identical layers
    two sub-layers 
    - multi-head self-attention mechanism
    = position-wise fully connected feed-forward network
    그 사이에 residual connection & layer normalization : LayerNorm(x + Sublayer(x))
    모든 sublayer들은(embedding layers포함) output dim = d_model = 512
    """
    def __init__(self, d_model=512, n_heads=8, dropout_p=0.1):
        super(EncoderBlock, self).__init__()
        self.multihead_attn = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.feed_forward = PositionwiseFeedForward(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
        self.dropout2 = nn.Dropout(p=dropout_p)
    def forward(self, embeddings, src_mask=None):
        # embedding shape : batch_size x seq_len x d_model
        # sublayer 1
        out1 = self.multihead_attn(q=embeddings, k=embeddings, v=embeddings, attn_mask=src_mask) # [batch_size, num_heads, seq_len, d_model]
        # Add & Norm 
        out1 = self.layer_norm1(self.dropout1(out1) + embeddings)
        # sublayer 2 
        out2 = self.feed_forward(out1) 
        # Add & Norm
        out2 = self.layer_norm2(self.dropout2(out2) + out1)
        return out2 # [batch, d_head, src_seq_len, d_model]
    

class DecoderBlock(nn.Module):
    """
    stack of n=6 identical layers
    two sub-layers as encoder
    + third sub-layer : multi-head attentoin over output of encoder stack

    modify self-attention sub-layer in decoder stack 
    (positions 뒤에 안보도록 masking)
    """
    def __init__(self, d_model=512, n_heads=8, dropout_p=0.1):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.masked_multihead_attn = MultiHeadAttention(self.d_model, n_heads) #CHECK 
        self.encdec_multihead_attn = MultiHeadAttention(self.d_model, n_heads)
        
        self.layer_norm1 = LayerNormalization(self.d_model)
        self.layer_norm2 = LayerNormalization(self.d_model)
        self.layer_norm3 = LayerNormalization(self.d_model)
        
        self.feed_forward = PositionwiseFeedForward(self.d_model)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)
    def forward(self, out_embedds, enc_embedds, src_mask=None, target_mask=None):
        # sublayer 1
        out1 = self.masked_multihead_attn(q=out_embedds, k=out_embedds, v=out_embedds, attn_mask=target_mask) # -> [batch, target seq_len, d_model] [64, 99, 512]
        # Add & Norm 
        out1 = self.layer_norm1(self.dropout1(out1) + out_embedds) 
        # sublayer 2
        # q : from out_embedds
        # k, v : from enc_embedds
        out2 = self.encdec_multihead_attn(q=out1, k=enc_embedds, v=enc_embedds, attn_mask=src_mask) # -> [batch, target seq_len, d_model] [64, 99, 512]
        # Add & Norm
        out2 = self.layer_norm2(self.dropout2(out2) + out1) 
        # sublayer 3
        out3 = self.feed_forward(out2) # -> [batch, target seq_len, d_model] [64, 99, 512]
        # Add & Norm
        out3 = self.layer_norm3(self.dropout3(out3) + out2)
        return out3 # [batch, d_head, tgt_seq_len, d_model] 
        
