import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from model.transformer import Transformer
from train import train
from dataset import generate_random_data 
from utils import load_yaml, count_parameters, initialize_weights, batchify_data, save_logs
import os 
import argparse
import numpy as np 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser("Implementation of Transformer in Pytorch")
    parser.add_argument("--output",
                        required=True,
                        type=str,
                        help="output path for the trained model")
    parser.add_argument("--log",
                        required=True,
                        type=str,
                        help="output path for saving the logs (including filename)")
    parser.add_argument("--cfg",
                        required=True,
                        type=str,
                        help="configuration path")
    return parser.parse_args()

    
def main():
    args = parse_args()
    log_save_path = args.log
    model_save_path = args.output 
    cfg = load_yaml(args.cfg)['train'] 
    
    os.makedirs(model_save_path, exist_ok=True)

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    print(f'[INFO] n_warmup: {cfg["warmup_steps"]} | max length : {cfg["max_len"]} | batch size : {cfg["batch_size"]} | epochs : {cfg["epochs"]} | lr : {cfg["learning_rate"]}')
    print(f'[INFO] d_model : {cfg["d_model"]} | n_heads : {cfg["n_heads"]} | n_layers : {cfg["n_layers"]} | d_ff : {cfg["d_ff"]} | dropout_p : {cfg["dropout_p"]}')
    
    print('[INFO] Load dataset ...')
    train_data = generate_random_data(20000, length=cfg['max_len'] - 2) # 10000
    val_data = generate_random_data(6000, length=cfg['max_len'] - 2)  # 3000

    train_loader = batchify_data(train_data, batch_size=cfg['batch_size'])
    val_loader = batchify_data(val_data, batch_size=cfg['batch_size'])

    print('[INFO] Load model ...')
    # sos, eos, padding, 3, 4
    model = Transformer(
        enc_vsize=5, 
        dec_vsize=5, 
        d_model=cfg['d_model'],
        max_len=cfg['max_len'],
        dropout_p=cfg['dropout_p'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'],
        device=device,
        src_pad_idx=2,
        tgt_pad_idx=2
    ).to(device)

    print(f'[INFO] # of trainable parameters : {count_parameters(model):,}') 
    model.apply(initialize_weights)

    criterion = nn.CrossEntropyLoss(ignore_index=2) 
    optimizer = optim.Adam(model.parameters(), 
                           betas=(0.9, 0.98), 
                           lr=cfg['learning_rate'], # default 0.001
                           eps=1e-9)

    def lr_scheduler(optimizer, warmup_steps, d_model):
        """equation (3)"""
        def lrate(step):
            return (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
        return LambdaLR(optimizer, lr_lambda=lrate)

    scheduler = lr_scheduler(optimizer, 
                             warmup_steps=cfg['warmup_steps'], 
                             d_model=cfg['d_model'])

    tr_losses, val_losses = train(model, train_loader, val_loader, 
                                  criterion, optimizer, scheduler,
                                  cfg['epochs'], device, model_save_path)
    
    save_logs(log_save_path, tr_losses, val_losses)
    print('[INFO] Successfully saved model!')

if __name__ == "__main__":
    main()

