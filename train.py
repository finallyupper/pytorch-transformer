import torch
from tqdm import tqdm
import os 
from typing import List, Tuple

def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          epochs=100, device=None, save_path='', best_val_loss = 0.60) -> Tuple[List[float], List[float]]:
    tr_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_tr_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            src, tgt = batch[:, 0], batch[:, 1]
            src, tgt = torch.tensor(src).to(device), torch.tensor(tgt).to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])  # tgt input excludes <eos>    # batch_size x tgt_seq_len x tgt_vocab_size
            output = output.reshape(-1, output.shape[-1]) # batch_size * tgt_seq_len x tgt_vocab_size  
            tgt = tgt[:, 1:].reshape(-1)  # tgt output excludes <sos>   
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_tr_loss += loss.item()
            
        if scheduler is not None:
            scheduler.step()

        tr_loss = total_tr_loss / len(train_loader)
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                src, tgt = batch[:, 0], batch[:, 1]
                src, tgt = torch.tensor(src, dtype=torch.long).to(device), torch.tensor(tgt, dtype=torch.long).to(device)
                output = model(src, tgt[:, :-1])
                output = output.reshape(-1, output.shape[-1])
                tgt = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt)
                total_val_loss += loss.item()


        val_loss = total_val_loss / len(val_loader)

        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f"[INFO] EPOCH {epoch+1}/{epochs}, Train Loss: {tr_loss}, Valid Loss: {val_loss}") 

        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            model_name = f'new_best_e{epoch+1}_{best_val_loss:0.5f}.pth'
            torch.save(model.state_dict(), os.path.join(save_path, model_name))
            print('[INFO] Successfully saved best model to', os.path.join(save_path, model_name))

    return tr_losses, val_losses
