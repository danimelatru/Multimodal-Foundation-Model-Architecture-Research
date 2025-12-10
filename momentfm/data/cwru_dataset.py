import os, re, pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy import interpolate
import scipy.io

def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True

def load_signal_from_mat(path):
    mat = scipy.io.loadmat(path)
    key_candidates = [k for k in mat.keys() if "_time" in k or "DE" in k]
    if not key_candidates:
        raise ValueError(f"No valid vibration key found in {path}")
    key = key_candidates[0]
    return np.squeeze(mat[key]).astype(np.float32)

def sliding_windows(signal, window=1024, stride=512):
    for start in range(0, len(signal) - window + 1, stride):
        yield signal[start:start + window]

class CWRU_dataset(Dataset):
    def __init__(self, config, phase="train"):
        self.phase = phase
        self.base_path = config.base_path
        self.cache_dir = config.cache_dir
        self.load_cache = getattr(config, "load_cache", True)
        self.seq_len = config.seq_len
        self.window = getattr(config, "window", 1024)
        self.stride = getattr(config, "stride", 512)

        # FIXED 4-CLASS LABEL MAP
        self.label_map = {"Normal": 0, "B": 1, "IR": 2, "OR": 3}

        cache_path = os.path.join(self.cache_dir, f"cwru_{phase}.pkl")
        make_dir_if_not_exists(self.cache_dir)

        if os.path.exists(cache_path) and self.load_cache:
            print("[INFO] Loading cached dataset...")
            with open(cache_path, "rb") as fp:
                self._timeseries, self._labels, self._loads = pickle.load(fp)
        else:
            print("[INFO] Building dataset from raw .mat files...")
            self._timeseries, self._labels, self._loads = self._build_dataset()
            with open(cache_path, "wb") as fp:
                pickle.dump((self._timeseries, self._labels, self._loads), fp)

        # Interpolate and normalize to fixed seq_len
        n_samples, timesteps = self._timeseries.shape
        f = interpolate.interp1d(np.linspace(0, 1, timesteps), self._timeseries)
        interp = f(np.linspace(0, 1, self.seq_len))
        mean = np.mean(interp, axis=-1, keepdims=True)
        std = np.std(interp, axis=-1, keepdims=True)
        self._timeseries = (interp - mean) / (std + 1e-8)
        self._timeseries = self._timeseries.reshape(n_samples, 1, self.seq_len)

        self._length = len(self._timeseries)

        # Make labels contiguous (0..n_classes-1) but still only from the 4 base classes
        unique_labels = np.unique(self._labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        self._labels = np.array([label_mapping[l] for l in self._labels])
        self.n_classes = len(unique_labels)
        self.num_classes = self.n_classes
        print(f"[INFO] Loaded {self._length} windows, {self.n_classes} contiguous classes.")
        print(f"[DEBUG] Label mapping used: {label_mapping}")

    # ----------------------------------------------------------------------
    def _build_dataset(self):
        """Load and window all .mat signals in the dataset folder."""
        X, y, loads = [], [], []

        for root, _, files in os.walk(self.base_path):
            for file in sorted(files):
                if not file.endswith(".mat"):
                    continue

                # Handle Normal_x.mat pattern
                if file.startswith("Normal"):
                    m = re.match(r'Normal_([0-3])\.mat', file)
                    if not m:
                        print(f"Skipping {file}: unrecognized normal pattern")
                        continue
                    fault = "Normal"
                    load = m.group(1)
                    size = None

                # Handle OR###@##_##.mat pattern (e.g. OR007@12_0.mat)
                elif file.startswith("OR"):
                    # Use original working pattern and then collapse to single "OR" class
                    m = re.match(r'(OR\d+)@(\d+)_([0-3])\.mat', file)
                    if not m:
                        print(f"Skipping {file}: unrecognized OR pattern")
                        continue
                    fault_full, speed, load = m.groups()
                    # Collapse all OR### variants into a single "OR" class
                    fault = "OR"

                # Handle legacy faulted pattern like B007_1.mat, IR014_3.mat, OR007_2.mat, etc.
                else:
                    m = re.match(r'([A-Z]+)(\d+)_([0-3])\.mat', file)
                    if not m:
                        print(f"Skipping {file}: unrecognized pattern")
                        continue
                    fault, size, load = m.groups()
                    # Here 'fault' will be 'B', 'IR', 'OR', etc.

                label_name = fault  # e.g. "Normal", "B", "IR", "OR"
                label = self.label_map.get(label_name, None)

                # STRICT: ONLY KEEP THE 4 BASE CLASSES
                if label is None:
                    print(f"Skipping {file}: fault type {label_name} not in base label_map")
                    continue

                # Load signal and create windows
                full_path = os.path.join(root, file)
                sig = load_signal_from_mat(full_path)
                for w in sliding_windows(sig, self.window, self.stride):
                    X.append(w)
                    y.append(label)
                    loads.append(int(load))

        if len(X) == 0:
            raise RuntimeError(
                f"No windows built from base_path={self.base_path}. "
                f"Check that the folder exists and contains .mat files with valid names."
            )

        return np.stack(X), np.array(y), np.array(loads)

    # ----------------------------------------------------------------------
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self._timeseries[idx], self._labels[idx]

    @property
    def timeseries(self):
        return self._timeseries

    @property
    def labels(self):
        return self._labels

    @property
    def loads(self):
        """Motor load labels (domain)"""
        return self._loads
