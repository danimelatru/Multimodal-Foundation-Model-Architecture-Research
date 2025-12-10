import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch import amp
from argparse import Namespace
from momentfm.data.cwru_dataset import CWRU_dataset
from momentfm.models.moment import MOMENT
from momentfm.common import TASKS

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
config = Namespace(
    base_path="/home/fernandeda/projects/CWRU_Dataset",
    cache_dir="/home/fernandeda/projects/moment/data/cache",
    window=1024,
    stride=512,
    seq_len=1024,
    task_name=TASKS.CLASSIFICATION,
)
dataset = CWRU_dataset(config)

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

print(f"[INFO] Dataset sizes -> Total: {total_len}, "
      f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------------------------------
# 2. Configure and initialize MOMENT model
# -----------------------------------------------------
config = Namespace(
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
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device being used: {device}")
model = MOMENT(config).to(device)
print(f"[INFO] Model initialized on {device}. Num classes: {dataset.num_classes}")

# Optional GPU sanity check
if torch.cuda.is_available():
    dummy = torch.randn(1, 1, 1024).to(device)
    with torch.no_grad():
        _ = model.classify(x_enc=dummy)
    print(f"[DEBUG] ✅ Forward pass successful on {torch.cuda.get_device_name(0)}")

# -----------------------------------------------------
# 3. Optionally freeze encoder
# -----------------------------------------------------
freeze_encoder = True
if freeze_encoder:
    for name, param in model.named_parameters():
        if not any(k in name.lower() for k in ["head", "classifier", "output"]):
            param.requires_grad = False
    print("[INFO] Encoder frozen; training head parameters only.")
else:
    print("[INFO] Full fine-tuning (encoder + classification head).")

# Debug: list trainable params
trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"[DEBUG] Trainable parameters: {len(trainable_params)}")
for n in trainable_params:
    print("   ->", n)

# -----------------------------------------------------
# 4. Training setup
# -----------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = amp.GradScaler("cuda")  # asume CUDA

checkpoint_path = "moment_cwru_final.pt"
if os.path.exists(checkpoint_path):
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

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

        with amp.autocast("cuda"):
            outputs = model.classify(x_enc=X.float(), input_mask=None)
            logits = outputs.logits.squeeze()
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
        print(f"[INFO] ✅ Checkpoint saved at epoch {epoch+1}")

# -----------------------------------------------------
# 6. Final save
# -----------------------------------------------------
torch.save(model.state_dict(), "moment_cwru_final.pt")
print("[INFO] 🧠 Final model saved in moment_cwru_final.pt ✅")