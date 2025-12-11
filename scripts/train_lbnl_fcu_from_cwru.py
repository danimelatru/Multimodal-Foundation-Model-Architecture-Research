# train_lbnl_fcu_from_cwru.py

import os
import numpy as np
import torch

from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from argparse import Namespace

from momentfm.data.lbnl_fcu_dataset import LBNL_FCU_Dataset
from momentfm.models.moment import MOMENT
from momentfm.common import TASKS


# -----------------------------------------------------
# 0. Hyperparameters
# -----------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42
LR = 3e-4
WEIGHT_DECAY = 1e-4

# Path to the LBNL CSV folder and cache
BASE_PATH = "/gpfs/workdir/fernandeda/projects/LBNL_FCU"
CACHE_DIR = "/gpfs/workdir/fernandeda/projects/moment/data/cache"

# Path to the CWRU fine-tuned checkpoint (4-class model)
CWRU_CKPT_PATH = "checkpoints/moment_cwru_finetuned.pt"

# Output checkpoint (LBNL fine-tuned model)
LBNL_CKPT_PATH = "checkpoints/moment_lbnl_from_cwru.pt"


# -----------------------------------------------------
# 1. Build LBNL dataset
# -----------------------------------------------------
print("[INFO] Loading LBNL FCU dataset...")

data_config = Namespace(
    base_path=BASE_PATH,
    cache_dir=CACHE_DIR,
    window=1024,
    stride=1024,  # no temporal overlap between windows
    seq_len=1024,
    task_name=TASKS.CLASSIFICATION,
    sensor_column="RM_TEMP",  # you can switch to another column later
    binary_labels=True,       # 0: fault-free (FaultFree.csv), 1: faulty (all others)
    load_cache=True,
)

full_dataset = LBNL_FCU_Dataset(data_config)

print(f"[INFO] LBNL dataset total samples: {len(full_dataset)}, "
      f"num_classes: {full_dataset.num_classes}")

# Reproducible splits (NOTE: this is window-level, not file-level)
g = torch.Generator().manual_seed(SEED)

total_len = len(full_dataset)
trainval_size = int(0.85 * total_len)
test_size = total_len - trainval_size

trainval_dataset, test_dataset = random_split(
    full_dataset, [trainval_size, test_size], generator=g
)

inner_train_size = int(0.85 * trainval_size)
val_size = trainval_size - inner_train_size

train_dataset, val_dataset = random_split(
    trainval_dataset, [inner_train_size, val_size], generator=g
)

print(
    f"[INFO] LBNL splits -> Train: {len(train_dataset)}, "
    f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
)

# -----------------------------------------------------
# 2. DataLoaders
# -----------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------------------------------
# 3. Configure MOMENT model (binary classification)
# -----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device being used: {device}")

model_config = Namespace(
    task_name=TASKS.CLASSIFICATION,
    n_channels=1,
    num_class=full_dataset.num_classes,  # should be 2 (fault-free vs faulty)
    seq_len=1024,
    patch_len=8,
    patch_stride_len=8,
    d_model=256,
    transformer_backbone="google/flan-t5-small",
    transformer_type="encoder_only",
    t5_config={"d_model": 256, "num_layers": 4, "num_heads": 8, "d_ff": 512},
    randomly_initialize_backbone=False,   # we will load CWRU weights
    freeze_embedder=False,
    freeze_encoder=False,                 # full fine-tuning on LBNL
    freeze_head=False,
    enable_gradient_checkpointing=True,
)

model = MOMENT(model_config).to(device)
print(f"[INFO] MOMENT model created. Num classes: {full_dataset.num_classes}")

# Sanity forward pass
if torch.cuda.is_available():
    dummy = torch.randn(1, 1, 1024).to(device)
    with torch.no_grad():
        _ = model.classify(x_enc=dummy)
    print(f"[DEBUG] ✅ Forward pass successful on {torch.cuda.get_device_name(0)}")

# -----------------------------------------------------
# 4. Load CWRU checkpoint (encoder) and discard old head
# -----------------------------------------------------
if os.path.exists(CWRU_CKPT_PATH):
    print(f"[INFO] Loading CWRU checkpoint from {CWRU_CKPT_PATH}")
    state_dict = torch.load(CWRU_CKPT_PATH, map_location=device)

    # Drop old 4-class head so we can re-initialize a 2-class head
    for key in list(state_dict.keys()):
        if key.startswith("head.linear."):
            print(f"[INFO] Dropping head parameter from checkpoint: {key}")
            del state_dict[key]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Missing keys (expected for new head): {missing}")
    print(f"[INFO] Unexpected keys: {unexpected}")
else:
    print("[WARN] CWRU checkpoint not found. Training LBNL model from random init.")

# -----------------------------------------------------
# 5. Loss (with class weighting) and optimizer
# -----------------------------------------------------

# Compute class weights on TRAIN subset to mitigate imbalance
train_indices = train_dataset.indices  # indices into full_dataset
train_labels = full_dataset.labels[train_indices]

class_counts = np.bincount(train_labels, minlength=full_dataset.num_classes)
print(f"[INFO] Train class counts: {class_counts}")

# Inverse-frequency weighting: weight_c ∝ 1 / count_c
class_weights = 1.0 / (class_counts + 1e-8)
class_weights = class_weights / class_weights.sum() * full_dataset.num_classes
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

print(f"[INFO] Class weights used in loss: {class_weights.cpu().numpy()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

use_amp = (device.type == "cuda")
scaler = GradScaler(enabled=use_amp)

os.makedirs(os.path.dirname(LBNL_CKPT_PATH), exist_ok=True)

# -----------------------------------------------------
# 6. Training + Validation
# -----------------------------------------------------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ------- TRAIN -------
    model.train()
    total_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(device).float(), y.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model.classify(x_enc=X, input_mask=None)
                logits = outputs.logits.squeeze()
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model.classify(x_enc=X, input_mask=None)
            logits = outputs.logits.squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_train_loss = total_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]

    # ------- VALIDATION -------
    model.eval()
    val_loss_total = 0.0
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device).float(), y.to(device)
            outputs = model.classify(x_enc=X)
            logits = outputs.logits.squeeze()
            v_loss = criterion(logits, y)
            val_loss_total += v_loss.item()

            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    avg_val_loss = val_loss_total / len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    print(
        f"[Epoch {epoch+1}/{EPOCHS}] "
        f"Train loss: {avg_train_loss:.4f} | "
        f"Val loss: {avg_val_loss:.4f} | "
        f"Val acc: {val_acc:.2f}% | "
        f"LR={current_lr:.6f}"
    )

    # Save best model according to validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), LBNL_CKPT_PATH)
        print(f"[INFO] ✅ New best model saved at epoch {epoch+1} "
              f"(Val acc = {val_acc:.2f}%)")

print(f"[INFO] Best validation accuracy achieved: {best_val_acc:.2f}%")


# -----------------------------------------------------
# 7. Final Test evaluation (single run)
# -----------------------------------------------------
# Reload best model before testing
if os.path.exists(LBNL_CKPT_PATH):
    state_dict = torch.load(LBNL_CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded best LBNL model from {LBNL_CKPT_PATH} for test evaluation.")

model.eval()
test_correct, test_total = 0, 0

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device).float(), y.to(device)
        outputs = model.classify(x_enc=X)
        logits = outputs.logits.squeeze()
        preds = torch.argmax(logits, dim=1)

        test_correct += (preds == y).sum().item()
        test_total += y.size(0)

test_acc = 100.0 * test_correct / test_total
print(f"[RESULT] ✅ Final LBNL test accuracy (fine-tuned from CWRU): {test_acc:.2f}%")
