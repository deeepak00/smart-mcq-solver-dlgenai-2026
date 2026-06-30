import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader

# Add relative paths for models/ and SCRATCH/ folders
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from utils import set_seed, softmax_np
from preprocess import BpeTokenizerScratch, precompute, MCQTensorDataset, IDX2LABEL
from scratch_model import ScratchMCQTransformer

warnings.filterwarnings('ignore')

CFG = {
    'folds': 5,
    'bs': 32,
    'max_len': 128,
    'd': 256,
    'h': 8,
    'layers': 4,
    'ff': 512,
    'drop': 0.15
}

def main():
    parser = argparse.ArgumentParser(description="Run Scratch MCQ Transformer inference.")
    parser.add_argument("--models_dir", type=str, default=None, help="Directory containing the model checkpoints and tokenizers")
    parser.add_argument("--submission_path", type=str, default="submission.csv", help="Output path for submission CSV")
    args = parser.parse_args()

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
    test_path = os.path.abspath(os.path.join(base_dir, '../../data/test.csv'))

    if not os.path.exists(test_path):
        # Fallback to kaggle path if local path is not present
        test_path = '/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv'

    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test dataset shape: {test_df.shape}")

    if args.models_dir is not None:
        models_dir = os.path.abspath(args.models_dir)
    else:
        models_dir = os.path.abspath(os.path.join(base_dir, '../../models'))


    test_p = np.zeros((len(test_df), 5))

    t_start = time.time()

    for fold in range(CFG['folds']):
        print(f"Running inference for Fold {fold+1}/{CFG['folds']}...")
        
        # Load the BPE tokenizer for the fold
        tok = BpeTokenizerScratch(max_vocab=20000)
        tok_path = os.path.join(models_dir, f"scratch_tokenizer_fold_{fold}.json")
        tok.load(tok_path)
        print(f"  Loaded tokenizer from {tok_path} (vocab size: {tok.size})")

        # Precompute inputs once
        te_ids, te_tids, te_mask, _ = precompute(test_df, tok, CFG['max_len'], has_labels=False)
        te_ld = DataLoader(MCQTensorDataset(te_ids, te_tids, te_mask),
                            CFG['bs'], shuffle=False, num_workers=0, pin_memory=USE_AMP)

        # Initialize model with vocabulary size of this fold's tokenizer
        model = ScratchMCQTransformer(
            vocab=tok.size,
            d=CFG['d'],
            h=CFG['h'],
            layers=CFG['layers'],
            ff=CFG['ff'],
            drop=CFG['drop']
        ).to(DEVICE)

        # Load weights
        model_path = os.path.join(models_dir, f"scratch_model_fold_{fold}.pt")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"  Loaded model weights from {model_path}")

        model.eval()
        tlog = []
        with torch.no_grad():
            for b in te_ld:
                ids  = b['input_ids'].to(DEVICE, non_blocking=USE_AMP)
                tids = b['token_type_ids'].to(DEVICE, non_blocking=USE_AMP)
                mask = b['attention_mask'].to(DEVICE, non_blocking=USE_AMP)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    logits = model(ids, tids, mask)
                tlog.append(logits.cpu().numpy())
        
        test_p += np.vstack(tlog) / CFG['folds']
        del model
        if USE_AMP:
            torch.cuda.empty_cache()

    print(f"Inference completed in {time.time()-t_start:.1f}s")

    test_probs = softmax_np(test_p)
    
    # Save test probabilities to disk
    np.save('scratch_test_probs.npy', test_probs)
    print("✅ scratch_test_probs.npy saved.")

    preds = [' '.join([IDX2LABEL[i] for i in np.argsort(p)[::-1][:3]]) for p in test_probs]
    sub = pd.DataFrame({'id': test_df['id'], 'Prediction': preds})
    sub.to_csv(args.submission_path, index=False)
    print(f"✅ Submission saved to {args.submission_path}!")

    # Fallback copy for Kaggle environment
    if os.path.exists('/kaggle/working'):
        import shutil
        target_sub = '/kaggle/working/submission.csv'
        target_probs = '/kaggle/working/scratch_test_probs.npy'
        
        if os.path.abspath(args.submission_path) != target_sub:
            try:
                shutil.copy(args.submission_path, target_sub)
                print(f"✅ Copied submission.csv to {target_sub} for Kaggle.")
            except Exception as e:
                print(f"⚠️ Failed to copy submission to {target_sub}: {e}")
                
        if os.path.abspath('scratch_test_probs.npy') != target_probs:
            try:
                shutil.copy('scratch_test_probs.npy', target_probs)
                print(f"✅ Copied scratch_test_probs.npy to {target_probs} for Kaggle.")
            except Exception as e:
                print(f"⚠️ Failed to copy scratch_test_probs.npy to {target_probs}: {e}")

if __name__ == "__main__":
    main()
