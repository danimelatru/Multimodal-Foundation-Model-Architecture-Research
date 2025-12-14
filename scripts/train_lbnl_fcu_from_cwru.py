import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from argparse import Namespace
from sklearn.metrics import confusion_matrix, classification_report

from momentfm.data.lbnl_fcu_dataset import LBNL_FCU_Dataset
from momentfm.models.moment import MOMENT
from momentfm.common import TASKS

plt.switch_backend('Agg')

# -----------------------------------------------------
# 0. Hyperparameters
# -----------------------------------------------------
BATCH_SIZE = 64
EPOCHS_WARMUP = 10  # Train only Head + Input Layer
EPOCHS_FULL = 40    # Train everything
SEED = 42
LR_WARMUP = 1e-3    # Higher LR to learn the new input projection quickly
LR_FULL = 1e-4      # Lower LR to preserve pre-trained knowledge

BASE_PATH = "/gpfs/workdir/fernandeda/projects/LBNL_FCU"
CACHE_DIR = "/gpfs/workdir/fernandeda/projects/moment/data/cache"
CWRU_CKPT_PATH = "checkpoints/moment_cwru_finetuned.pt"
LBNL_CKPT_PATH = "checkpoints/moment_lbnl_LPFT.pt"
CONFUSION_MATRIX_PATH = "lbnl_confusion_matrix_LPFT.png"

# -----------------------------------------------------
# 1. Multi-Channel Dataset
# -----------------------------------------------------
print("[INFO] Loading LBNL FCU dataset (Multi-Channel)...")
SELECTED_COLUMNS = ['RM_TEMP', 'FCU_DAT', 'FCU_CVLV', 'FCU_HVLV']

data_config = Namespace(
    base_path=BASE_PATH,
    cache_dir=CACHE_DIR,
    window=1024,
    stride=1024,
    seq_len=1024,
    task_name=TASKS.CLASSIFICATION,
    sensor_column=SELECTED_COLUMNS,
    binary_labels=True,
    load_cache=True,
)
full_dataset = LBNL_FCU_Dataset(data_config)

g = torch.Generator().manual_seed(SEED)
total_len = len(full_dataset)
trainval_size = int(0.85 * total_len)
test_size = total_len - trainval_size
trainval_dataset, test_dataset = random_split(full_dataset, [trainval_size, test_size], generator=g)

inner_train_size = int(0.85 * trainval_size)
val_size = trainval_size - inner_train_size
train_dataset, val_dataset = random_split(trainval_dataset, [inner_train_size, val_size], generator=g)

# -----------------------------------------------------
# 2. Sampler (Balanced)
# -----------------------------------------------------
print("[INFO] Configuring Sampler...")
train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
class_counts = np.bincount(train_labels, minlength=full_dataset.num_classes)
weights_np = 1.0 / (class_counts + 1e-8)
# Slight tweak to favor precision again
weights_np[0] = weights_np[0] * 0.8 
sample_weights = weights_np[train_labels]
sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------------------------------
# 3. Model Setup
# -----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = Namespace(
    task_name=TASKS.CLASSIFICATION,
    n_channels=len(SELECTED_COLUMNS),
    num_class=2,
    seq_len=1024,
    patch_len=8,
    patch_stride_len=8,
    d_model=256,
    transformer_backbone="google/flan-t5-small",
    transformer_type="encoder_only",
    t5_config={"d_model": 256, "num_layers": 4, "num_heads": 8, "d_ff": 512},
    randomly_initialize_backbone=False,
    freeze_embedder=False,
    freeze_encoder=False,
    freeze_head=False,
    enable_gradient_checkpointing=True,
)
model = MOMENT(model_config).to(device)

if os.path.exists(CWRU_CKPT_PATH):
    print(f"[INFO] Loading CWRU Transformer Body...")
    state_dict = torch.load(CWRU_CKPT_PATH, map_location=device)
    for key in list(state_dict.keys()):
        if "head" in key or "patch_embedding" in key:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)

criterion = nn.CrossEntropyLoss()
scaler = GradScaler(enabled=(device.type == "cuda"))
best_score = 0.0

# -----------------------------------------------------
# 4. Phase 1: WARMUP (Freeze Backbone)
# -----------------------------------------------------
print(f"\n[INFO] Starting PHASE 1: Warmup for {EPOCHS_WARMUP} epochs (Backbone Frozen)")

# Freeze Transformer
for param in model.parameters():
    param.requires_grad = False
# Unfreeze Input Layer (Patch Embedding) and Head
for param in model.patch_embedding.parameters():
    param.requires_grad = True
for param in model.head.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_WARMUP)

for epoch in range(EPOCHS_WARMUP):
    model.train()
    total_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device).float(), y.to(device)
        optimizer.zero_grad()
        with autocast(enabled=(device.type == "cuda")):
            outputs = model.classify(x_enc=X)
            loss = criterion(outputs.logits.squeeze(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    # Validation (Optional during warmup, but good to see progress)
    print(f"[Warmup {epoch+1}/{EPOCHS_WARMUP}] Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------------------------------
# 5. Phase 2: FULL TRAINING (Unfreeze All)
# -----------------------------------------------------
print(f"\n[INFO] Starting PHASE 2: Full Finetuning for {EPOCHS_FULL} epochs (Unfrozen)")

# Unfreeze Everything
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)

for epoch in range(EPOCHS_FULL):
    model.train()
    total_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device).float(), y.to(device)
        optimizer.zero_grad()
        with autocast(enabled=(device.type == "cuda")):
            outputs = model.classify(x_enc=X)
            loss = criterion(outputs.logits.squeeze(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    scheduler.step()

    # Validation
    model.eval()
    all_val_preds, all_val_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device).float(), y.to(device)
            outputs = model.classify(x_enc=X)
            preds = torch.argmax(outputs.logits.squeeze(), dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_targets.extend(y.cpu().numpy())

    report = classification_report(all_val_targets, all_val_preds, output_dict=True, zero_division=0)
    f1_class_0 = report['0']['f1-score']
    val_acc = report['accuracy']
    
    print(f"[Epoch {epoch+1}/{EPOCHS_FULL}] Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f} | F1 (Cls 0): {f1_class_0:.4f}")

    if f1_class_0 > best_score:
        best_score = f1_class_0
        torch.save(model.state_dict(), LBNL_CKPT_PATH)
        print(f"[INFO] ✅ Saved Best Model (F1: {best_score:.4f})")

# -----------------------------------------------------
# 6. Final Evaluation
# -----------------------------------------------------
if os.path.exists(LBNL_CKPT_PATH):
    model.load_state_dict(torch.load(LBNL_CKPT_PATH, map_location=device))

model.eval()
all_test_preds, all_test_targets = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device).float(), y.to(device)
        outputs = model.classify(x_enc=X)
        preds = torch.argmax(outputs.logits.squeeze(), dim=1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_targets.extend(y.cpu().numpy())

print("\n" + "="*50)
print(classification_report(all_test_targets, all_test_preds, target_names=['Fault-Free', 'Faulty'], digits=4))

cm = confusion_matrix(all_test_targets, all_test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fault-Free', 'Faulty'], yticklabels=['Fault-Free', 'Faulty'])
plt.title(f'Confusion Matrix (LP-FT Strategy)')
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH)