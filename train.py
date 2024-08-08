from model.transformer import Transformer 
import torch 
import torch.nn as nn 
import yaml 
import torch.optim as optim
from tqdm import tqdm 
# data: https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    cfg = load_yaml('/home/yoojinoh/Others/pytorch-transformer/data/config.yaml')['train']
    n_heads = cfg['n_heads']
    source_vocab_size = cfg['source_vocab_size']
    target_vocab_size = cfg['target_vocab_size']
    max_seq_length = cfg['max_seq_length']
    batch_size = 64 

    transformer = Transformer(source_vocab_size, target_vocab_size, n_heads)

    # Generate random sample data
    src_data = torch.randint(1, source_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length) [64, 100]
    tgt_data = torch.randint(1, target_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length) [64, 100]

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()
    
    for epoch in (range(100)):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1]) # [64, 99, 5000] 
        loss = criterion(output.contiguous().view(-1, target_vocab_size), tgt_data[:, 1:].contiguous().view(-1)) # preds = [64 * 99, 5000], answer = [64, 99]
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}/100, Loss: {loss.item()}")

if __name__ == "__main__":
    main()