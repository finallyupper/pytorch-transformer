import torch 
import torch.nn as nn
import torch.nn.functional as F
from .attention import PositionalEncoding 
from .block import EncoderBlock, DecoderBlock 

class Transformer(nn.Module):
    """
    input embedding - p.e - enc block(xN) 
    - output embedding - p.e - dec block(xN) - linear - softmax (output probabilities)

    """
    def __init__(self, enc_vsize, dec_vsize, n_blocks=6, d_model=512, dropout_p=0.1):
        super(Transformer, self).__init__()
        self.enc_emb = nn.Embedding(num_embeddings=enc_vsize, embedding_dim=d_model)
        self.dec_emb = nn.Embedding(num_embeddings=dec_vsize, embedding_dim=d_model)

        self.n_blocks = n_blocks
        self.d_model = d_model 
        self.positional_encoding = PositionalEncoding(d_model)

        self.enc_blocks = nn.ModuleList(
            [EncoderBlock(d_model) for _ in range(n_blocks)]
        )
        self.dec_blocks = nn.ModuleList(
            [DecoderBlock(d_model) for _ in range(n_blocks)]
        )
        self.fc = nn.Linear(d_model, dec_vsize)
        self.dropout = nn.Dropout(dropout_p)

    def make_src_mask(self, source, pad_idx = 0):
        """
        Padding mask : padding token이 model prediction에 영향을 미치지 않도록 하기 위함.
        ex.
        vocab:
            1 -> apple
            2 -> orange
            3 -> banana
            0 -> padding 
        sent 1 = ['apple', 'orange', 'padding', 'padding']    
        sent 2 = ['banana', 'padding', 'padding', 'padding']
        padding index = 0

        source sequence = [[1, 2, 0, 0], [3, 0, 0, 0]]     ->  batch_size x seq_len
        source mask = [[[[True, True, False, False]], [[True, False, False, False]]]]  -> batch_size x 1 x "1" x seq_le
            
        """
        src_mask = (source != pad_idx).unsqueeze(1).unsqueeze(2) #  batch_size x seq_len -> batch_size x 1 x 1 x seq_len
        return src_mask 
    
    def make_target_mask(self, target, pad_idx = 0):
        """
        Padding mask : padding token이 model prediction에 영향을 미치지 않도록 하기 위함.
        Attention mask : 현재 단어가 미래 단어를 예측할수 없도록함.
        ex.
        vocab:
            1 -> apple
            2 -> orange
            3 -> banana
            0 -> padding 
        sent 1 = ['apple', 'orange', 'padding', 'padding']    
        sent 2 = ['banana', 'padding', 'padding', 'padding']
        padding index = 0
        
        target :
        1) padding mask
            target sequence = [[1, 2, 0, 0], [3, 0, 0, 0]]     ->  batch_size x seq_len
            target mask = [[[[True, True, False, False]], [[True, False, False, False]]]]  -> batch_size x 1 x "seq_len" x seq_len
        2) attention mask
            = [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1]
            ]
         e.g. sent1 ['apple', _, _, _] => ['apple', 'orange', _, _] => ...
            
        """
        # emb shape: batch_size x seq_len
        padding_mask = (target != pad_idx).unsqueeze(1).unsqueeze(3) #  batch_size x seq_len -> batch_size x 1 x seq_len x seq_len

        target_seq_len = target.size(1)
        
        # attn_mask = torch.tril(torch.ones((target_seq_len, target_seq_len)), diagonal=0).bool() # seq_len x seq_len
        nopeak_mask = (1 - torch.triu(torch.ones(1, target_seq_len, target_seq_len), diagonal=1)).bool()
       
        target_mask = nopeak_mask & padding_mask
        
        return target_mask 
         
    def forward(self, enc_input, dec_input):
        src_mask = self.make_src_mask(enc_input) # [batch, 1, 1, src_seq_len]
        target_mask = self.make_target_mask(dec_input) # [batch, 1, tgt_seq_len, tgt_seq_len]

        input_embedding = self.enc_emb(enc_input) # [batch, src_seq_len, d_model] [64, 100, 512]
        output_embedding = self.dec_emb(dec_input) # [batch, tgt_seq_len, d_model] [64, 99, 512]

        enc_out = self.dropout(self.positional_encoding(input_embedding)) # [batch, src_seq_len, d_model]
        for block in self.enc_blocks:
            enc_out = block(enc_out, src_mask = src_mask)
        
        dec_out = self.dropout(self.positional_encoding(output_embedding))
        for block in self.dec_blocks:
            dec_out = block(dec_out, enc_out, src_mask = src_mask, target_mask = target_mask) #  [batch, tgt_seq_len, d_model]

        out = self.fc(dec_out) # [batch, tgt_seq_len, dec_vsize] [64, 99, 5000]
        return torch.softmax(out, dim=-1)