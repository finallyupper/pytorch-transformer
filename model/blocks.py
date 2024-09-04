import torch
from torch import nn
from model.attention import MultiHeadAttention 
from model.ffn import PositionwiseFeedForward
from model.layer_norm import LayerNormalization

class Encoderlayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_p = 0.1) -> None:
        super(Encoderlayer, self).__init__()
        self.multihead_attn = MultiHeadAttention(d_model=d_model,
                                                 n_heads=n_heads,
                                                 dropout_p=dropout_p)
        self.layernorm1 = LayerNormalization(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.ffn = PositionwiseFeedForward(d_model=d_model,
                                           d_ff=d_ff,
                                           dropout_p=dropout_p)
        self.layernorm2 = LayerNormalization(d_model=d_model)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, embedding, mask) -> torch.Tensor:
        # embedding shape : batch_size x src_seq_len x d_model
        out1 = self.multihead_attn(Q=embedding, K=embedding, V=embedding, mask=mask)
        out1 = self.dropout1(out1)
        out1 = self.layernorm1(embedding + out1) 

        out2 = self.ffn(out1)
        out2 = self.dropout2(out2)
        out2 = self.layernorm2(out1 + out2) 
    
        return out2  # batch_size x src_seq_len x d_model
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.masked_multihead_attn = MultiHeadAttention(d_model=d_model,
                                                        n_heads=n_heads,
                                                        dropout_p=dropout_p)
        self.layernorm1 = LayerNormalization(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        
        self.multihead_attn = MultiHeadAttention(d_model=d_model,
                                                 n_heads=n_heads,
                                                 dropout_p=dropout_p)
        self.layernorm2 = LayerNormalization(d_model=d_model)
        self.dropout2 = nn.Dropout(dropout_p)


        self.ffn = PositionwiseFeedForward(d_model=d_model,
                                           d_ff=d_ff,
                                           dropout_p=dropout_p)
        
        
        self.layernorm3 = LayerNormalization(d_model=d_model)
        self.dropout3 = nn.Dropout(dropout_p)

    def forward(self, enc_emb, tgt_emb, src_mask, tgt_mask) -> torch.Tensor:
        out1 = self.masked_multihead_attn(Q=tgt_emb, K=tgt_emb, V=tgt_emb, mask=tgt_mask)
        out1 = self.dropout1(out1)
        out1 = self.layernorm1(tgt_emb + out1)
        

        out2 = self.multihead_attn(Q=out1, K=enc_emb, V=enc_emb, mask=src_mask)
        out2 = self.dropout2(out2)
        out2 = self.layernorm2(out1 + out2)

        out3 = self.ffn(out2)
        out3 = self.dropout3(out3)
        out3 = self.layernorm3(out2 + out3)

        return out3

