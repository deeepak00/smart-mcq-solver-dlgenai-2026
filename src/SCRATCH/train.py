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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
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


def training_model(model, optimizer, train, val, epochs, patience, loss_fn, scheduler, device, fold, models_dir, use_amp=False, scaler=None, scheduler_step_per_epoch=False):
    """
    Generic training loop for any PyTorch model.
    Handles inputs of type: single tensor, list/tuple of tensors, or dictionary of tensors.
    Supports early stopping based on MAP@3, checkpoints, confusion matrices, and Wandb logging.
    """
    patience_counter = 0
    best_val = 0.0
    best_val_preds = []
    best_val_targets = []
    
    for e in range(epochs):
        if patience_counter > patience:
            print(f"Early stopping triggered")
            break
        
        # training loop
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_targets = []
        
        for i, batch in enumerate(tqdm(train, desc=f"Epoch {e} Train")):           
            # Generic batch unpacking
            if isinstance(batch, dict):
                label_key = next((k for k in ['labels', 'label', 'target', 'targets', 'y'] if k in batch), None)
                if label_key is not None:
                    y = batch[label_key].to(device)
                    x = {k: v.to(device) for k, v in batch.items() if k != label_key}
                else:
                    x = {k: v.to(device) for k, v in batch.items()}
                    y = None
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                y = y.to(device)
                if isinstance(x, dict):
                    x = {k: v.to(device) for k, v in x.items()}
                elif isinstance(x, (list, tuple)):
                    x = [item.to(device) for item in x]
                else:
                    x = x.to(device)
            else:
                x = batch.to(device)
                y = None

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Forward pass
                if isinstance(x, dict):
                    preds = model(**x)
                elif isinstance(x, (list, tuple)):
                    preds = model(*x)
                else:
                    preds = model(x)
                    
                loss = loss_fn(preds, y)
                
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Step scheduler if per-batch
            if scheduler is not None and not scheduler_step_per_epoch and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                
            train_losses.append(loss.item())
            _, predicted_classes = torch.max(preds, 1) 
            all_train_preds.extend(predicted_classes.cpu().numpy())
            all_train_targets.extend(y.cpu().numpy() if y is not None else [])

        train_acc = accuracy_score(all_train_targets, all_train_preds) if len(all_train_targets) > 0 else 0.0
        train_f1 = f1_score(all_train_targets, all_train_preds, average='macro') if len(all_train_targets) > 0 else 0.0
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch: {e}, Training_f1_macro: {train_f1:.4f}, Training_accuracy: {train_acc:.4f}, Avg_train_loss: {avg_train_loss:.4f}")

        # validation loop
        model.eval()
        all_val_preds = []
        all_val_targets = []
        all_val_logits = []
        val_losses = []
        
        with torch.inference_mode(): 
            for batch in val:
                # Generic batch unpacking
                if isinstance(batch, dict):
                    label_key = next((k for k in ['labels', 'label', 'target', 'targets', 'y'] if k in batch), None)
                    if label_key is not None:
                        y = batch[label_key].to(device)
                        x = {k: v.to(device) for k, v in batch.items() if k != label_key}
                    else:
                        x = {k: v.to(device) for k, v in batch.items()}
                        y = None
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                    y = y.to(device)
                    if isinstance(x, dict):
                        x = {k: v.to(device) for k, v in x.items()}
                    elif isinstance(x, (list, tuple)):
                        x = [item.to(device) for item in x]
                    else:
                        x = x.to(device)
                else:
                    x = batch.to(device)
                    y = None
                
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    if isinstance(x, dict):
                        preds = model(**x)
                    elif isinstance(x, (list, tuple)):
                        preds = model(*x)
                    else:
                        preds = model(x)
                        
                    loss = loss_fn(preds, y)
                val_losses.append(loss.item())
                all_val_logits.append(preds.cpu().numpy())
                
                _, predicted_classes = torch.max(preds, 1)
                all_val_preds.extend(predicted_classes.cpu().numpy())
                all_val_targets.extend(y.cpu().numpy() if y is not None else [])
                
        val_acc = accuracy_score(all_val_targets, all_val_preds) if len(all_val_targets) > 0 else 0.0
        val_f1 = f1_score(all_val_targets, all_val_preds, average='macro') if len(all_val_targets) > 0 else 0.0
        avg_val_loss = np.mean(val_losses)
        
        # Calculate MAP@3
        vlog = np.vstack(all_val_logits)
        val_map3 = map3(all_val_targets, softmax_np(vlog)) if len(all_val_targets) > 0 else 0.0
        
        # Step scheduler if per-epoch or ReduceLROnPlateau
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_map3)        
            elif scheduler_step_per_epoch:
                scheduler.step()
            
        print(f"Epoch: {e}, validation_map3: {val_map3:.4f}, validation_accuracy: {val_acc:.4f}, Avg_val_loss: {avg_val_loss:.4f}")

        # early stopping and model checkpointing based on MAP@3
        checkpoint_path = os.path.join(models_dir, f"scratch_model_fold_{fold}.pt")
        if val_map3 > best_val:
            best_val = val_map3
            patience_counter = 0 
            torch.save(model.state_dict(), checkpoint_path)
            best_val_preds = all_val_preds
            best_val_targets = all_val_targets
            print(f"  ✅ Best validation MAP@3 updated: {best_val:.4f}. Checkpoint saved.")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
            
        # Log metrics to wandb
        if wandb.run is not None:
            wandb.log({
                f"fold_{fold+1}/train_accuracy": train_acc,
                f"fold_{fold+1}/train_f1_macro": train_f1,
                f"fold_{fold+1}/train_loss": avg_train_loss,
                f"fold_{fold+1}/val_accuracy": val_acc,
                f"fold_{fold+1}/val_map3": val_map3,
                f"fold_{fold+1}/val_loss": avg_val_loss,
                "epoch": e
            })
            
    # Load best weights
    checkpoint_path = os.path.join(models_dir, f"scratch_model_fold_{fold}.pt")
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Validation report and plot
    if len(best_val_targets) > 0:
        print(f"\nClassification Report for Fold {fold+1}:")
        print(classification_report(best_val_targets, best_val_preds, target_names=OPTIONS))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(best_val_targets, best_val_preds), annot=True, fmt='d', cmap='Blues',
                    xticklabels=OPTIONS, yticklabels=OPTIONS)
        plt.title(f"Confusion Matrix - Fold {fold+1}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig(os.path.join(models_dir, f"confusion_matrix_fold_{fold}.png"))
        plt.close()
    
    return model


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
        
        loss_fn = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])

        model = training_model(
            model=model,
            optimizer=opt,
            train=tr_ld,
            val=vl_ld,
            epochs=CFG['epochs'],
            patience=CFG['patience'],
            loss_fn=loss_fn,
            scheduler=sched,
            device=DEVICE,
            fold=fold,
            models_dir=models_dir,
            use_amp=USE_AMP,
            scaler=scaler,
            scheduler_step_per_epoch=False
        )

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
