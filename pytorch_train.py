import torch
import torch.nn as nn
import yaml
import torch.optim as optim
from tqdm import tqdm

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

class TransformerModel(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model, n_heads, n_layers, d_ff, dropout_p):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(source_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(target_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers, dim_feedforward=d_ff, dropout=dropout_p)
        self.fc_out = nn.Linear(d_model, target_vocab_size)
        
    def forward(self, src, tgt):
        src = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        src = src.permute(1, 0, 2)  # (batch_size, seq_length, d_model) -> (seq_length, batch_size, d_model)
        tgt = tgt.permute(1, 0, 2)  # (batch_size, seq_length, d_model) -> (seq_length, batch_size, d_model)
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

def main():
    cfg = load_yaml('/home/yoojinoh/Others/pytorch-transformer/data/config.yaml')['train']
    source_vocab_size = cfg['source_vocab_size']
    target_vocab_size = cfg['target_vocab_size']
    d_model = cfg['d_model']
    n_heads = cfg['n_heads']
    n_layers = cfg['n_layers']
    d_ff = cfg['d_ff']
    dropout_p = cfg['dropout_p']
    max_seq_length = cfg['max_seq_length']
    batch_size = 64

    model = TransformerModel(source_vocab_size, target_vocab_size, d_model, n_heads, n_layers, d_ff, dropout_p)

    # Generate random sample data
    src_data = torch.randint(1, source_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, target_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model.train()
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(src_data, tgt_data[:, :-1])  # [seq_length, batch_size, target_vocab_size]
        loss = criterion(output.view(-1, target_vocab_size), tgt_data[:, 1:].contiguous().view(-1))  # preds = [64 * 99, 5000], answer = [64, 99]
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}/100, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
