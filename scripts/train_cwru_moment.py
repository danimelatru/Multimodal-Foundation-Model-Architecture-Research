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

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scaler = amp.GradScaler("cuda")  # ✅ new AMP API

checkpoint_path = "moment_cwru_final.pt"
if os.path.exists(checkpoint_path):
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

# -----------------------------------------------------
# 5. Training + Evaluation
# -----------------------------------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device).float(), y.to(device)
        optimizer.zero_grad()

        # 🚀 Mixed precision training
        with amp.autocast("cuda"):
            outputs = model.classify(x_enc=X.float(), input_mask=None)
            logits = outputs.logits.squeeze()
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"[Epoch {epoch+1}/{epochs}] Training loss: {avg_loss:.4f} | LR={current_lr:.6f}")

    # Periodic evaluation
    if (epoch + 1) % 5 == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model.classify(x_enc=X.float())
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = 100 * correct / total
        print(f"[INFO] 🧩 Test Accuracy after {epoch+1} epochs: {acc:.2f}%")

        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[INFO] ✅ Checkpoint saved at epoch {epoch+1}")

# -----------------------------------------------------
# 6. Final save
# -----------------------------------------------------
torch.save(model.state_dict(), "moment_cwru_final.pt")
print("[INFO] 🧠 Final model saved in moment_cwru_final.pt ✅")

