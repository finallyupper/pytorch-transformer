import torch
from model.transformer import Transformer
from utils import load_yaml 
import argparse 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser("Implementation of Transformer in Pytorch")
    parser.add_argument("--model",
                        required=True,
                        type=str,
                        help="path of the trained model")
    parser.add_argument("--cfg",
                        required=True,
                        type=str,
                        help="configuration path")
    return parser.parse_args()

def predict(model, input_sequence, max_length=10, SOS_token=0, EOS_token=1, PAD_token=2, device=None)->list:
    model.eval()

    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    for _ in range(max_length):
        pred = model(input_sequence, y_input)
        next_item = pred.topk(1)[1].view(-1)[-1].item() 
        next_item = torch.tensor([[next_item]], device=device)

        y_input = torch.cat((y_input, next_item), dim=1)
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()

def main():
    args = parse_args()
    model_path = args.model
    cfg = load_yaml(args.cfg)['test'] 
    
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

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

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    #################################### EDIT HERE ####################################
    examples = [
        torch.tensor([[0, 3, 3, 3, 3, 3, 3, 3, 3, 1]], dtype=torch.long, device=device),
        torch.tensor([[0, 4, 4, 4, 4, 4, 4, 4, 4, 1]], dtype=torch.long, device=device),
        torch.tensor([[0, 3, 4, 3, 4, 3, 4, 3, 4, 1]], dtype=torch.long, device=device),
        torch.tensor([[0, 4, 3, 4, 3, 4, 3, 4, 3, 1]], dtype=torch.long, device=device),
        torch.tensor([[0, 3, 4, 3, 1]], dtype=torch.long, device=device),
    ]
    ###################################################################################
    for idx, example in enumerate(examples):
        result = predict(model, example, max_length=10, device=device)
        print(f"Example {idx}")
        print(f"Input: {example.view(-1).tolist()[1:-1]}")
        print(f"Continuation: {result[1:-1]}")
        print()


if __name__ == "__main__":
    main()