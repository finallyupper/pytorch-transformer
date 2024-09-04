import torch.nn as nn
import pandas as pd
import numpy as np 
import yaml 
import matplotlib.pyplot as plt 
import argparse

def load_data(file_path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m) -> None:
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data) # updated

def batchify_data(data, batch_size=16, padding=False, padding_token=-1) -> list: 
    """
    Returns dummy data in a batch
    reference : https://wikidocs.net/156986
    """
    batches = []
    for idx in range(0, len(data), batch_size):
        if idx + batch_size < len(data):
            if padding:
                max_batch_length = 0
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"[INFO] {len(batches)} batches of size {batch_size}")

    return batches

def save_logs(save_path, train_losses, val_losses) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Over Epochs")
    plt.savefig(save_path) 
    print(f'[INFO] Loss graph saved at {save_path}')