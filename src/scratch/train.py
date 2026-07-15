import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import sys
import os

# Add project root and src/scratch directory to sys.path for robust imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import CFG, set_seed, get_device, softmax_np, map3, warmup_cosine
from preprocessed import BpeTokenizerScratch, texts_from, precompute, MCQTensorDataset
from models.scratch import ScratchMCQModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True,
                    help="Directory containing train.csv and test.csv")
    p.add_argument('--output_dir', type=str, default='/kaggle/working/artifacts',
                    help="Where to save tokenizers, model checkpoints, and OOF/test logits")
    p.add_argument('--folds', type=int, default=CFG['folds'])
    p.add_argument('--epochs', type=int, default=CFG['epochs'])
    p.add_argument('--bs', type=int, default=CFG['bs'])
    p.add_argument('--lr', type=float, default=CFG['lr'])
    p.add_argument('--seed', type=int, default=CFG['seed'])
    p.add_argument('--use_wandb', action='store_true',
                    help="Flag to enable logging to weights and biases")
    p.add_argument('--wandb_project', type=str, default='smart-mcq-solver-scratch',
                    help="Weights and biases project name")
    p.add_argument('--wandb_run_name', type=str, default=None,
                    help="Weights and biases run name")
    return p.parse_args()


def main():
    args = parse_args()

    # allow CLI overrides of the shared CFG
    CFG['folds'] = args.folds
    CFG['epochs'] = args.epochs
    CFG['bs'] = args.bs
    CFG['lr'] = args.lr
    CFG['seed'] = args.seed

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(CFG['seed'])
    device, use_amp = get_device()

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=CFG
        )

    train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(args.data_path, 'test.csv'))

    from utils import LABEL2IDX
    train_df['answer'] = train_df['answer'].map(LABEL2IDX)

    skf = StratifiedKFold(CFG['folds'], shuffle=True, random_state=CFG['seed'])
    y = train_df['answer'].values

    oof = np.zeros((len(train_df), 5))
    test_p = np.zeros((len(test_df), 5))

    for fold, (tri, vli) in enumerate(skf.split(train_df, y)):
        print(f"\n========== FOLD {fold + 1}/{CFG['folds']} ==========")
        fold_dir = os.path.join(args.output_dir, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        tok = BpeTokenizerScratch(max_vocab=CFG['max_vocab'])
        tok.build(texts_from(train_df.iloc[tri]))
        print("Tokenizer size:", tok.size)

        tr_ids, tr_tids, tr_mask, tr_lbl = precompute(train_df.iloc[tri], tok, CFG['max_len'], has_labels=True)
        vl_ids, vl_tids, vl_mask, vl_lbl = precompute(train_df.iloc[vli], tok, CFG['max_len'], has_labels=True)
        te_ids, te_tids, te_mask, _ = precompute(test_df, tok, CFG['max_len'], has_labels=False)

        tr_ld = DataLoader(MCQTensorDataset(tr_ids, tr_tids, tr_mask, tr_lbl),
                            batch_size=CFG['bs'], shuffle=True, num_workers=0, pin_memory=use_amp)
        vl_ld = DataLoader(MCQTensorDataset(vl_ids, vl_tids, vl_mask, vl_lbl),
                            batch_size=CFG['bs'], shuffle=False, num_workers=0, pin_memory=use_amp)
        te_ld = DataLoader(MCQTensorDataset(te_ids, te_tids, te_mask),
                            batch_size=CFG['bs'], shuffle=False, num_workers=0, pin_memory=use_amp)

        model = ScratchMCQModel(
            vocab_size=tok.size,
            d_model=CFG['d_model'],
            conv_ch=CFG['conv_ch'],
            hidden=CFG['hidden'],
            drop=CFG['drop']
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
        scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)

        total_steps = CFG['epochs'] * len(tr_ld)
        warmup_steps = int(CFG['warmup_frac'] * total_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda s: warmup_cosine(s, warmup_steps, total_steps)
        )

        best, best_state, wait = 0.0, None, 0

        for ep in range(CFG['epochs']):
            model.train()
            tl = 0.0

            for b in tr_ld:
                ids = b['input_ids'].to(device)
                tids = b['token_type_ids'].to(device)
                mask = b['attention_mask'].to(device)
                lbl = b['labels'].to(device)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(ids, tids, mask)
                    loss = F.cross_entropy(logits, lbl, label_smoothing=CFG['label_smoothing'])

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

                tl += loss.item()

            model.eval()
            vlog, vlbl = [], []
            val_loss_sum = 0.0
            with torch.no_grad():
                for b in vl_ld:
                    ids = b['input_ids'].to(device)
                    tids = b['token_type_ids'].to(device)
                    mask = b['attention_mask'].to(device)
                    lbl = b['labels'].to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        logits = model(ids, tids, mask)
                        loss = F.cross_entropy(logits, lbl, label_smoothing=CFG['label_smoothing'])
                    val_loss_sum += loss.item()
                    vlog.append(logits.cpu().numpy())
                    vlbl.append(lbl.cpu().numpy())

            vlog = np.vstack(vlog)
            vlbl = np.concatenate(vlbl)
            vm3 = map3(vlbl, softmax_np(vlog))
            vacc = (vlog.argmax(1) == vlbl).mean()
            val_loss = val_loss_sum / len(vl_ld)
            print(f"Ep{ep + 1:02d} | loss={tl / len(tr_ld):.4f} | val_loss={val_loss:.4f} | acc={vacc:.4f} | map3={vm3:.4f}")

            if args.use_wandb:
                import wandb
                wandb.log({
                    f"fold{fold}/epoch": ep + 1,
                    f"fold{fold}/train_loss": tl / len(tr_ld),
                    f"fold{fold}/val_loss": val_loss,
                    f"fold{fold}/val_acc": vacc,
                    f"fold{fold}/val_map3": vm3,
                    f"fold{fold}/lr": opt.param_groups[0]['lr'],
                })

            if vm3 > best:
                best = vm3
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= CFG['patience']:
                    print("Early stopping.")
                    break

        model.load_state_dict(best_state)
        model.eval()

        # --- save this fold's artifacts (tokenizer + model + metadata) ---
        tok.save(os.path.join(fold_dir, 'tokenizer.json'))
        torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pt'))
        with open(os.path.join(fold_dir, 'fold_meta.json'), 'w') as f:
            json.dump({
                'vocab_size': tok.size,
                'd_model': CFG['d_model'],
                'conv_ch': CFG['conv_ch'],
                'hidden': CFG['hidden'],
                'drop': CFG['drop'],
                'max_len': CFG['max_len'],
                'best_map3': float(best),
            }, f, indent=2)
        print(f"Saved fold {fold} artifacts to {fold_dir}")

        if args.use_wandb:
            import wandb
            wandb.log({
                f"fold{fold}/best_val_map3": best
            })

        # OOF
        vpred = []
        with torch.no_grad():
            for b in vl_ld:
                ids = b['input_ids'].to(device)
                tids = b['token_type_ids'].to(device)
                mask = b['attention_mask'].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(ids, tids, mask)
                vpred.append(logits.cpu().numpy())
        oof[vli] = np.vstack(vpred)

        # TEST (kept for reference/ensembling; inference.py recomputes this
        # cleanly from saved artifacts for use in a separate notebook)
        tpred = []
        with torch.no_grad():
            for b in te_ld:
                ids = b['input_ids'].to(device)
                tids = b['token_type_ids'].to(device)
                mask = b['attention_mask'].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(ids, tids, mask)
                tpred.append(logits.cpu().numpy())
        test_p += np.vstack(tpred) / CFG['folds']

        del model, opt, scaler, tr_ld, vl_ld, te_ld
        torch.cuda.empty_cache()

    oof_probs = softmax_np(oof)
    oof_map3 = map3(y, oof_probs)
    print("OOF MAP@3:", oof_map3)

    if args.use_wandb:
        import wandb
        wandb.log({"oof_map3": oof_map3})
        wandb.finish()

    np.save(os.path.join(args.output_dir, 'oof_logits.npy'), oof)
    np.save(os.path.join(args.output_dir, 'test_logits.npy'), test_p)
    with open(os.path.join(args.output_dir, 'cfg.json'), 'w') as f:
        json.dump(CFG, f, indent=2)

    print(f"\nAll fold artifacts + oof/test logits saved under: {args.output_dir}")


if __name__ == '__main__':
    main()