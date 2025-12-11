import os
import torch
import numpy as np
from argparse import Namespace
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import confusion_matrix, classification_report

from momentfm.data.lbnl_fcu_dataset import LBNL_FCU_Dataset
from momentfm.models.moment import MOMENT
from momentfm.common import TASKS

# ---------------------------------------------------------------------
# 0. Hyperparameters and paths
# ---------------------------------------------------------------------
BATCH_SIZE = 64
SEED = 42

BASE_PATH = "/gpfs/workdir/fernandeda/projects/LBNL_FCU"
CACHE_DIR = "/gpfs/workdir/fernandeda/projects/moment/data/cache"
CKPT_PATH = "checkpoints/moment_lbnl_from_cwru.pt"

# ---------------------------------------------------------------------
# 1. Build LBNL FCU dataset (same config as for fine-tuning)
# ---------------------------------------------------------------------
print("[INFO] Loading LBNL FCU dataset...")

data_config = Namespace(
    base_path=BASE_PATH,          # folder with CSVs
    cache_dir=CACHE_DIR,
    window=1024,
    stride=1024,
    seq_len=1024,
    task_name=TASKS.CLASSIFICATION,
    sensor_column="RM_TEMP",      # same sensor used during training
    binary_labels=True,           # 0: fault-free, 1: faulty
    load_cache=True,
)

full_dataset = LBNL_FCU_Dataset(data_config)

print(
    f"[INFO] LBNL FCU -> total samples: {len(full_dataset)}, "
    f"num_classes (reported): {full_dataset.num_classes}"
)

# ---------------------------------------------------------------------
# 2. Rebuild the same train/val/test split as in fine-tuning script
#    (85% train+val, 15% test, with SEED = 42)
# ---------------------------------------------------------------------
g = torch.Generator().manual_seed(SEED)

total_len = len(full_dataset)
trainval_size = int(0.85 * total_len)
test_size = total_len - trainval_size

trainval_dataset, test_dataset = random_split(
    full_dataset, [trainval_size, test_size], generator=g
)

# You could further split trainval into train/val, but here we only
# care about test_dataset for evaluation.
print(
    f"[INFO] LBNL splits -> Train+Val: {len(trainval_dataset)}, "
    f"Test: {len(test_dataset)}"
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------
# 3. Build MOMENT model and load fine-tuned LBNL checkpoint
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device being used: {device}")

NUM_CLASSES = full_dataset.num_classes  # should be 2 for binary_labels=True

model_config = Namespace(
    task_name=TASKS.CLASSIFICATION,
    n_channels=1,
    num_class=NUM_CLASSES,        # 2 outputs: [normal, faulty]
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

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

print(f"[INFO] Loading fine-tuned LBNL checkpoint from {CKPT_PATH}")
state_dict = torch.load(CKPT_PATH, map_location=device)

# IMPORTANT: do NOT drop the head parameters here.
# This checkpoint is already fine-tuned on LBNL with the correct 2-class head.
missing, unexpected = model.load_state_dict(state_dict, strict=True)
print(f"[INFO] State dict loaded (strict=True). Missing: {missing}, Unexpected: {unexpected}")

model.eval()

# ---------------------------------------------------------------------
# 4. Evaluation on TEST set + confusion matrix
# ---------------------------------------------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device).float()   # [batch, 1, seq_len]
        y = y.to(device)

        outputs = model.classify(x_enc=X)
        logits = outputs.logits  # [batch, NUM_CLASSES]
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Accuracy
accuracy = (all_preds == all_labels).mean() * 100.0

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)

print(f"[RESULT] ✅ LBNL TEST accuracy (from moment_lbnl_from_cwru.pt): {accuracy:.2f}%")
print(f"[RESULT] True label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
print(f"[RESULT] Pred label distribution: {dict(zip(*np.unique(all_preds, return_counts=True)))}")

print("[RESULT] Confusion matrix (rows = true, cols = predicted):")
print(cm)

print("\n[RESULT] Classification report:")
print(report)