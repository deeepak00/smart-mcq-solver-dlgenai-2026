import sys
import os
import time
import math
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import wandb

# Add relative paths for models/ and SCRATCH/ folders
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from utils import set_seed, softmax_np, map3, warmup_cosine
from preprocess import BpeTokenizerScratch, texts_from, precompute, MCQTensorDataset, OPTIONS, LABEL2IDX, IDX2LABEL
from scratch_model import ScratchMCQTransformer

warnings.filterwarnings('ignore')

CFG = {
    'folds': 5,
    'epochs': 10,
    'bs': 32,
    'lr': 3e-4,
    'wd': 0.01,
    'd': 256,
    'h': 8,
    'layers': 4,
    'ff': 512,
    'max_len': 128,
    'drop': 0.15,
    'patience': 3,
    'label_smoothing': 0.1,
    'warmup_frac': 0.1,
    'use_wandb': True,
    'project': 'smart-mcq-solver-scratch',
    'run_name': 'scratch-model-5fold'
}

def main():
    parser = argparse.ArgumentParser(description="Train Scratch MCQ Transformer model with CV.")
    parser.add_argument("--epochs", type=int, default=CFG['epochs'], help="Number of training epochs")
    parser.add_argument("--use_wandb", action="store_true", default=CFG['use_wandb'], help="Enable logging to Weights & Biases")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb", help="Disable logging to Weights & Biases")
    args = parser.parse_args()
    
    CFG['epochs'] = args.epochs
    CFG['use_wandb'] = args.use_wandb

    set_seed(42)
    DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = (DEVICE.type == 'cuda')

    if USE_AMP:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected — running on CPU.")

    # Paths Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.abspath(os.path.join(base_dir, '../../data/train.csv'))
    test_path = os.path.abspath(os.path.join(base_dir, '../../data/test.csv'))

    if not os.path.exists(train_path):
        # Fallback to kaggle path if local path is not present
        train_path = '/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv'
        test_path = '/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv'

    print(f"Loading data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Train dataset shape: {train_df.shape}")

    # Initialize wandb
    if CFG['use_wandb']:
        wandb.init(
            project=CFG['project'],
            name=CFG['run_name'],
            config=CFG
        )

    # Cross-Validation Initialization
    skf    = StratifiedKFold(CFG['folds'], shuffle=True, random_state=42)
    y      = train_df['answer'].map(LABEL2IDX).values
    oof    = np.zeros((len(train_df), 5))

    # Make sure models dir exists
    models_dir = os.path.abspath(os.path.join(base_dir, '../../models'))
    os.makedirs(models_dir, exist_ok=True)

    t_start = time.time()

    for fold, (tri, vli) in enumerate(skf.split(train_df, y)):
        print(f"\n{'='*40}\nFOLD {fold+1}/{CFG['folds']}\n{'='*40}")
        t_fold = time.time()

        # Train a BPE Tokenizer from scratch strictly on current train fold texts
        tok = BpeTokenizerScratch(max_vocab=20000)
        tok.build(texts_from(train_df.iloc[tri]))
        print(f"  Fold BPE vocab size: {tok.size}")

        # Save the BPE tokenizer for inference
        tok_path = os.path.join(models_dir, f"scratch_tokenizer_fold_{fold}.json")
        tok.save(tok_path)
        print(f"  Saved tokenizer to {tok_path}")

        # Precompute inputs once
        tr_ids, tr_tids, tr_mask, tr_lbl = precompute(train_df.iloc[tri], tok, CFG['max_len'])
        vl_ids, vl_tids, vl_mask, vl_lbl = precompute(train_df.iloc[vli], tok, CFG['max_len'])

        tr_ld = DataLoader(MCQTensorDataset(tr_ids, tr_tids, tr_mask, tr_lbl),
                            CFG['bs'], shuffle=True,  num_workers=0, pin_memory=USE_AMP)
        vl_ld = DataLoader(MCQTensorDataset(vl_ids, vl_tids, vl_mask, vl_lbl),
                            CFG['bs'], shuffle=False, num_workers=0, pin_memory=USE_AMP)

        # Initialize model
        model = ScratchMCQTransformer(
            vocab=tok.size,
            d=CFG['d'],
            h=CFG['h'],
            layers=CFG['layers'],
            ff=CFG['ff'],
            drop=CFG['drop']
        ).to(DEVICE)

        opt    = torch.optim.AdamW(model.parameters(), CFG['lr'], weight_decay=CFG['wd'])
        scaler = torch.amp.GradScaler(device='cuda', enabled=USE_AMP)
        
        total_steps  = CFG['epochs'] * len(tr_ld)
        warmup_steps = int(CFG['warmup_frac'] * total_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda s: warmup_cosine(s, warmup_steps, total_steps)
        )

        best, best_state, wait = 0.0, None, 0

        for ep in range(CFG['epochs']):
            # Train Step
            model.train()
            tl, tc, tt = 0.0, 0, 0
            for b in tr_ld:
                ids  = b['input_ids'].to(DEVICE, non_blocking=USE_AMP)
                tids = b['token_type_ids'].to(DEVICE, non_blocking=USE_AMP)
                mask = b['attention_mask'].to(DEVICE, non_blocking=USE_AMP)
                lbl  = b['labels'].to(DEVICE, non_blocking=USE_AMP)
                
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    logits = model(ids, tids, mask)
                    loss = F.cross_entropy(logits, lbl, label_smoothing=CFG['label_smoothing'])
                    
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                
                tl += loss.item()
                tc += (logits.argmax(-1) == lbl).sum().item()
                tt += lbl.size(0)

            # Validation Step
            model.eval()
            vlog, vlbl = [], []
            with torch.no_grad():
                for b in vl_ld:
                    ids  = b['input_ids'].to(DEVICE, non_blocking=USE_AMP)
                    tids = b['token_type_ids'].to(DEVICE, non_blocking=USE_AMP)
                    mask = b['attention_mask'].to(DEVICE, non_blocking=USE_AMP)
                    lbl  = b['labels'].to(DEVICE, non_blocking=USE_AMP)
                    
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                        logits = model(ids, tids, mask)
                    vlog.append(logits.cpu().numpy())
                    vlbl.append(lbl.cpu().numpy())

            vlog, vlbl = np.vstack(vlog), np.concatenate(vlbl)
            vacc = (vlog.argmax(1) == vlbl).mean()
            vm3  = map3(vlbl, softmax_np(vlog))
            
            epoch_loss = tl / len(tr_ld)
            epoch_acc = tc / tt
            print(f"Ep{ep+1:02d}: loss={epoch_loss:.3f} acc={epoch_acc:.3f} | val_acc={vacc:.3f} map3={vm3:.3f}")

            # Log to wandb
            if CFG['use_wandb']:
                wandb.log({
                    f"fold_{fold+1}/train_loss": epoch_loss,
                    f"fold_{fold+1}/train_acc": epoch_acc,
                    f"fold_{fold+1}/val_acc": vacc,
                    f"fold_{fold+1}/val_map3": vm3,
                    f"fold_{fold+1}/lr": sched.get_last_lr()[0],
                    "epoch": ep + 1
                })

            # Early Stopping check
            if vm3 > best:
                best = vm3
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
                print(f"  ✅ Best MAP@3 Updated: {best:.4f}")
            else:
                wait += 1
                if wait >= CFG['patience']:
                    print(f"  Early stopping triggered after {CFG['patience']} epochs without improvement.")
                    break

        # Restore best weights and predict OOF
        model.load_state_dict(best_state)
        model.eval()

        # Save model checkpoint
        model_path = os.path.join(models_dir, f"scratch_model_fold_{fold}.pt")
        torch.save(best_state, model_path)
        print(f"  Saved fold {fold} model state dict to {model_path}")

        vlog = []
        with torch.no_grad():
            for b in vl_ld:
                ids  = b['input_ids'].to(DEVICE, non_blocking=USE_AMP)
                tids = b['token_type_ids'].to(DEVICE, non_blocking=USE_AMP)
                mask = b['attention_mask'].to(DEVICE, non_blocking=USE_AMP)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    logits = model(ids, tids, mask)
                vlog.append(logits.cpu().numpy())
        oof[vli] = np.vstack(vlog)

        print(f"  Fold time: {time.time()-t_fold:.1f}s")
        
        # Log best MAP@3 for the fold
        if CFG['use_wandb']:
            wandb.run.summary[f"fold_{fold+1}_best_map3"] = best

        del model
        if USE_AMP:
            torch.cuda.empty_cache()

    print(f"\nTotal training time: {time.time()-t_start:.1f}s")

    oof_probs = softmax_np(oof)
    oof_map3 = map3(y, oof_probs)
    print(f"Final Honest OOF MAP@3: {oof_map3:.4f}")

    # Save OOF probs to disk
    np.save('scratch_oof_probs.npy', oof_probs)
    print("✅ scratch_oof_probs.npy saved.")

    if CFG['use_wandb']:
        wandb.run.summary["oof_map3"] = oof_map3
        wandb.finish()

if __name__ == "__main__":
    main()
