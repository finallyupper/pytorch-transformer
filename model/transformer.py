import torch
import torch.nn as nn
from model.blocks import Encoderlayer, DecoderLayer
from model.embedding import TransformerEmbedding

class Encoder(nn.Module):
    """
    Encoder module of the Transformer architecture.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the input sequences.
        dropout_p (float): Dropout probability.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        d_ff (int): Dimension of the feed-forward layer.
        device (torch.device): Device to run the model on.
    """
    def __init__(self, vocab_size, d_model, max_len, 
                 dropout_p=0.1, n_heads=8, n_layers=6, d_ff=2048, device=None) -> None:
        super(Encoder,self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, # Token embedding + Positional Embedding
                                              d_model=d_model,
                                              max_len=max_len,
                                              dropout_p=dropout_p,
                                              device=device)
        self.enc_layers = nn.ModuleList(
            [Encoderlayer(d_model=d_model,
                          n_heads=n_heads,
                          d_ff=d_ff,
                          dropout_p=dropout_p) for _ in range(n_layers)])
    
    def forward(self, x, mask) -> torch.Tensor:
        out = self.embedding(x)
        for enc_layer in self.enc_layers:
            out = enc_layer(out, mask)
        return out 
    

class Decoder(nn.Module):
    """
    Decoder module of the Transformer architecture.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the input sequences.
        dropout_p (float): Dropout probability.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of decoder layers.
        d_ff (int): Dimension of the feed-forward layer.
        device (torch.device): Device to run the model on.
    """
    def __init__(self, vocab_size, d_model, max_len, 
                 dropout_p=0.1, n_heads=8, n_layers=6, d_ff=2048,device=None) -> None:
        super(Decoder,self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, # Token embedding + Positional Embedding
                                              d_model=d_model,
                                              max_len=max_len,
                                              dropout_p=dropout_p,
                                              device=device)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model,
                          n_heads=n_heads,
                          d_ff=d_ff,
                          dropout_p=dropout_p) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, enc_emb, tgt, src_mask, tgt_mask) -> torch.Tensor:
        # enc_emb shape : batch_size x src_seq_len x d_model
        # tgt shape : batch_size x tgt_seq_len
        tgt_emb = self.embedding(tgt) # batch_size x tgt_seq_len x d_model
        for dec_layer in self.dec_layers:
            tgt_emb = dec_layer(enc_emb, tgt_emb, src_mask, tgt_mask)
        out = self.linear(tgt_emb) # batch_size x tgt_seq_len x tgt_vocab_size
        return out
        
class Transformer(nn.Module):
    def __init__(self, enc_vsize, dec_vsize, 
                 d_model, 
                 max_len, 
                 dropout_p=0.1, 
                 n_heads=8, 
                 n_layers=6, 
                 d_ff=2048, 
                 device=None,
                 src_pad_idx=0, 
                 tgt_pad_idx=0) -> None:
        super(Transformer, self).__init__()
        self.device = device 

        self.encoder = Encoder(vocab_size=enc_vsize,
                               d_model=d_model,
                               max_len=max_len,
                               dropout_p=dropout_p,
                               n_heads=n_heads,
                               n_layers=n_layers,
                               d_ff=d_ff,
                               device=device)
        
        self.decoder = Decoder(vocab_size=dec_vsize,
                               d_model=d_model,
                               max_len = max_len,
                               dropout_p=dropout_p,
                               n_heads=n_heads,
                               n_layers=n_layers,
                               d_ff=d_ff,
                               device=device)
        self.src_pad_idx = src_pad_idx 
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_mask(self, source) -> torch.Tensor:
        """Padding mask"""
        src_mask = (source != self.src_pad_idx).unsqueeze(1).unsqueeze(2) #  batch_size x seq_len -> batch_size x 1 x 1 x seq_len
        return src_mask 
    
    def make_target_mask(self, target) -> torch.Tensor:
        """
        1) padding mask - finds padding token and assigns False
        2) attention mask (target mask) - limits access available parts  
        """
        padding_mask = (target != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        target_seq_len = target.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, target_seq_len, target_seq_len), diagonal=1)).bool().to(self.device)
        target_mask = nopeak_mask & padding_mask
        
        return target_mask 
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src) # batch_size x 1 x 1 x src_seq_len
        tgt_mask = self.make_target_mask(tgt) # batch_size x 1 x 1 x tgt_seq_len

        enc_emb = self.encoder(src, src_mask) # batch_size x src_seq_len x d_model
        tgt_emb = self.decoder(enc_emb, tgt, src_mask, tgt_mask) # batch_size x tgt_seq_len x tgt_vocab_size
        return tgt_emb # No softmax as applied in CrossEntroyLoss