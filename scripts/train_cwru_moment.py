import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from argparse import Namespace

from momentfm.data.cwru_dataset import CWRU_dataset
from momentfm.models.moment import MOMENT
from momentfm.common import TASKS

from torch.cuda.amp import GradScaler, autocast

# -----------------------------------------------------
# 0. Hyperparameters
# -----------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

# -----------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------
print("[INFO] Loading CWRU dataset...")
data_config = Namespace(
    base_path="/gpfs/workdir/fernandeda/projects/CWRU_Dataset",
    cache_dir="/gpfs/workdir/fernandeda/projects/moment/data/cache",
    window=1024,
    stride=1024,
    seq_len=1024,
    task_name=TASKS.CLASSIFICATION,
    load_cache=True,
)
dataset = CWRU_dataset(data_config)

print("Unique labels:", np.unique(dataset.labels))
print("Number of classes declared:", dataset.num_classes)

# Reproducible splits
g = torch.Generator().manual_seed(SEED)

# 85% train+val, 15% test
total_len = len(dataset)
trainval_size = int(0.85 * total_len)
test_size = total_len - trainval_size

trainval_dataset, test_dataset = random_split(
    dataset, [trainval_size, test_size], generator=g
)

# Of the 85% train+val: 85% train, 15% val
inner_train_size = int(0.85 * trainval_size)
val_size = trainval_size - inner_train_size

train_dataset, val_dataset = random_split(
    trainval_dataset, [inner_train_size, val_size], generator=g
)

print(
    f"[INFO] Dataset sizes -> Total: {total_len}, "
    f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------------------------------
# 2. Configure and initialize MOMENT model (PRETRAINED + FULL FT)
# -----------------------------------------------------
model_config = Namespace(
    task_name=TASKS.CLASSIFICATION,
    n_channels=1,
    num_class=dataset.num_classes,
    seq_len=1024,
    patch_len=8,
    patch_stride_len=8,
    d_model=256,
    transformer_backbone="google/flan-t5-small",
    transformer_type="encoder_only",
    t5_config={"d_model": 256, "num_layers": 4, "num_heads": 8, "d_ff": 512},

    # Use pretrained backbone
    randomly_initialize_backbone=False,

    # full fine-tuning
    freeze_embedder=False,
    freeze_encoder=False,
    freeze_head=False,

    # to save memory
    enable_gradient_checkpointing=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device being used: {device}")
model = MOMENT(model_config).to(device)
print(f"[INFO] Model initialized on {device}. Num classes: {dataset.num_classes}")

if torch.cuda.is_available():
    dummy = torch.randn(1, 1, 1024).to(device)
    with torch.no_grad():
        _ = model.classify(x_enc=dummy)
    print(f"[DEBUG] ✅ Forward pass successful on {torch.cuda.get_device_name(0)}")

# Debug: how many params are actually trainable
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] Trainable parameters: {n_trainable}")

# -----------------------------------------------------
# 4. Training setup
# -----------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

use_amp = (device.type == "cuda")
scaler = GradScaler(enabled=use_amp)

# === CHECKPOINTS CONFIG ===
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_path = os.path.join(CHECKPOINT_DIR, "moment_cwru_pretrained.pt")

if os.path.exists(checkpoint_path):
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # If the head shape changed (e.g. num_classes), drop old head params
    for key in list(state_dict.keys()):
        if key.startswith("head.linear."):
            print(f"[INFO] Dropping head parameter from checkpoint: {key}")
            del state_dict[key]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint with missing keys: {missing}")
    print(f"[INFO] Loaded checkpoint with unexpected keys: {unexpected}")
else:
    print("[INFO] No checkpoint found, training from scratch (pretrained backbone only).")

# -----------------------------------------------------
# 5. Training + Validation + Test
# -----------------------------------------------------
for epoch in range(EPOCHS):
    # ------- TRAIN -------
    model.train()
    total_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(device).float(), y.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model.classify(x_enc=X.float(), input_mask=None)
                logits = outputs.logits.squeeze()
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model.classify(x_enc=X.float(), input_mask=None)
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
            outputs = model.classify(x_enc=X.float())
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

    # ------- PERIODIC TEST EVAL + CHECKPOINT -------
    if (epoch + 1) % 5 == 0:
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device).float(), y.to(device)
                outputs = model.classify(x_enc=X.float())
                logits = outputs.logits.squeeze()
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == y).sum().item()
                test_total += y.size(0)

        test_acc = 100.0 * test_correct / test_total
        print(f"[INFO] 🧩 Test Accuracy after {epoch+1} epochs: {test_acc:.2f}%")

        torch.save(model.state_dict(), checkpoint_path)
        print(f"[INFO] ✅ Checkpoint (pretrained FT) saved at epoch {epoch+1}")

# -----------------------------------------------------
# 6. Final save
# -----------------------------------------------------
torch.save(model.state_dict(), checkpoint_path)
print(f"[INFO] 🧠 Final (pretrained FT) model saved in {checkpoint_path} ✅")
