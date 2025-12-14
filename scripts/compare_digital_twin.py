import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold  # <--- Corrected Import
from sklearn.metrics import accuracy_score
from argparse import Namespace

from momentfm.data.digital_twin_dataset import DigitalTwinDataset
from momentfm.models.moment import MOMENT
from momentfm.common import TASKS

# --- CONFIGURATION ---
FILE_PATH = "my_dataset_test.mat"
CWRU_CKPT = "checkpoints/moment_cwru_finetuned.pt" 
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_type):
    # Base Configuration
    model_config = Namespace(
        task_name=TASKS.CLASSIFICATION,
        n_channels=6,  # 6 Robot Axes
        num_class=9,   # 9 Fault Classes
        seq_len=512,
        patch_len=8,
        patch_stride_len=8,
        d_model=256,
        transformer_backbone="google/flan-t5-small",
        transformer_type="encoder_only",
        t5_config={"d_model": 256, "num_layers": 4, "num_heads": 8, "d_ff": 512},
        randomly_initialize_backbone=False, # Standard loads pre-trained MOMENT
        freeze_embedder=False, freeze_encoder=False, freeze_head=False, enable_gradient_checkpointing=True
    )
    model = MOMENT(model_config).to(device)
    
    if model_type == "CWRU":
        if os.path.exists(CWRU_CKPT):
            # print(f"   [CWRU] Loading checkpoint: {CWRU_CKPT}")
            state_dict = torch.load(CWRU_CKPT, map_location=device)
            
            # Clean dictionary: Remove Head (4 vs 9 classes) and Input (1 vs 6 channels)
            clean_dict = {}
            for k, v in state_dict.items():
                if "head" not in k and "patch_embedding" not in k:
                    clean_dict[k] = v
                    
            model.load_state_dict(clean_dict, strict=False)
        else:
            print(f"   [WARN] CWRU Checkpoint not found at {CWRU_CKPT}")

    # Freeze Transformer Backbone (Linear Probing)
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze Input Projection (Adapter) and Classification Head
    for param in model.patch_embedding.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
        
    return model

def train_one_fold(model, train_loader, test_loader):
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model.classify(x_enc=X)
            loss = criterion(out.logits.squeeze(), y)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model.classify(x_enc=X)
            p = torch.argmax(out.logits.squeeze(), dim=1)
            preds.extend(p.cpu().numpy())
            targets.extend(y.cpu().numpy())
            
    return accuracy_score(targets, preds)

# --- EXPERIMENT ---
print("[INFO] Initializing Digital Twin Dataset...")
dataset = DigitalTwinDataset(FILE_PATH, seq_len=512)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

results_std = []
results_cwru = []

print("\n" + "="*60)
print(f"{'Fold':<5} | {'Standard MOMENT':<20} | {'CWRU-Finetuned':<20}")
print("="*60)

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    train_sub = Subset(dataset, train_idx)
    test_sub = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_sub, batch_size=BATCH_SIZE, shuffle=False)
    
    # 1. Evaluate Standard MOMENT
    model_std = get_model("Standard")
    acc_std = train_one_fold(model_std, train_loader, test_loader)
    results_std.append(acc_std)
    
    # 2. Evaluate CWRU-Finetuned MOMENT
    model_cwru = get_model("CWRU")
    acc_cwru = train_one_fold(model_cwru, train_loader, test_loader)
    results_cwru.append(acc_cwru)
    
    print(f"{fold+1:<5} | {acc_std:.4f}               | {acc_cwru:.4f}")

print("-" * 60)
print(f"AVG   | {np.mean(results_std):.4f}               | {np.mean(results_cwru):.4f}")
print("="*60)

if np.mean(results_cwru) > np.mean(results_std):
    print("\n[CONCLUSION] CWRU Pre-training IMPROVED transfer learning on Robotics data!")
else:
    print("\n[CONCLUSION] CWRU Pre-training did NOT help (or hurt) compared to Base MOMENT.")