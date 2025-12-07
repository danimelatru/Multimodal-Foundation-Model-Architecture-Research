from momentfm.data import CWRU_dataset

class Config:
    base_path = "/home/fernandeda/projects/CWRU_Dataset"
    cache_dir = "/home/fernandeda/projects/moment/cache"
    seq_len = 1024
    window = 1024
    stride = 512
    load_cache = False

config = Config()
dataset = CWRU_dataset(config, phase="train")

print(f"Dataset length: {len(dataset)}")
print(f"Number of classes: {dataset.n_classes}")
print(f"Example shape: {dataset[0][0].shape}")
