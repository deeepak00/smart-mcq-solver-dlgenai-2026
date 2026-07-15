import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
import os

# Add project root and src/scratch directory to sys.path for robust imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import IDX2LABEL, get_device
from preprocessed import BpeTokenizerScratch, precompute, MCQTensorDataset
from models.scratch import ScratchMCQModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--artifacts_dir', type=str, required=True)
    p.add_argument('--output_path', type=str, default='/kaggle/working/submission.csv')
    p.add_argument('--bs', type=int, default=32)
    return p.parse_args()


def load_fold(fold_dir, device):
    with open(os.path.join(fold_dir, 'fold_meta.json')) as f:
        meta = json.load(f)

    tok = BpeTokenizerScratch.load(os.path.join(fold_dir, 'tokenizer.json'))

    model = ScratchMCQModel(
        vocab_size=meta['vocab_size'],
        d_model=meta['d_model'],
        conv_ch=meta['conv_ch'],
        hidden=meta['hidden'],
        drop=meta['drop'],
    ).to(device)
    state = torch.load(os.path.join(fold_dir, 'model.pt'), map_location=device)
    model.load_state_dict(state)
    model.eval()

    return tok, model, meta


def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    args = parse_args()
    device, use_amp = get_device()

    test_df = pd.read_csv(os.path.join(args.data_path, 'test.csv'))

    fold_dirs = sorted(glob.glob(os.path.join(args.artifacts_dir, 'fold*')))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold* directories found under {args.artifacts_dir}")
    print(f"Found {len(fold_dirs)} fold artifacts: {fold_dirs}")

    test_p = np.zeros((len(test_df), 5))

    for fold_dir in fold_dirs:
        print(f"\n--- Inference for {fold_dir} ---")
        tok, model, meta = load_fold(fold_dir, device)

        te_ids, te_tids, te_mask, _ = precompute(test_df, tok, meta['max_len'], has_labels=False)
        te_ld = DataLoader(MCQTensorDataset(te_ids, te_tids, te_mask),
                            batch_size=args.bs, shuffle=False, num_workers=0)

        tpred = []
        with torch.no_grad():
            for b in te_ld:
                ids = b['input_ids'].to(device)
                tids = b['token_type_ids'].to(device)
                mask = b['attention_mask'].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(ids, tids, mask)
                tpred.append(logits.cpu().numpy())
        test_p += np.vstack(tpred) / len(fold_dirs)

        del model
        torch.cuda.empty_cache()

    test_probs = softmax_np(test_p)
    preds = [' '.join([IDX2LABEL[i] for i in np.argsort(p)[::-1][:3]]) for p in test_probs]

    sub = pd.DataFrame({
        'ID': test_df['id'] if 'id' in test_df.columns else np.arange(1, len(test_df) + 1),
        'Prediction': preds
    })
    sub.to_csv(args.output_path, index=False)
    print(f"\nSaved submission to {args.output_path}")
    print(sub.head())


if __name__ == '__main__':
    main()