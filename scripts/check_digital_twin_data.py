import scipy.io
import numpy as np

file_path = 'my_dataset_test.mat'

try:
    mat = scipy.io.loadmat(file_path)
    print("KEYS:", mat.keys())
    
    for key, val in mat.items():
        if isinstance(val, np.ndarray):
            print(f"Key: {key} | Shape: {val.shape} | Type: {val.dtype}")
            
except Exception as e:
    print(f"Error loading .mat file: {e}")