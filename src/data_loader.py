import numpy as np
import os

def load_nclt_scan(bin_path):
    data = np.fromfile(bin_path, dtype=np.float32)
    data = data.reshape(-1, 4)
    return data[:, :3]

def scan_generator(scan_dir):
    files = sorted(os.listdir(scan_dir))
    for fname in files:
        if fname.endswith(".bin"):
            yield fname, load_nclt_scan(os.path.join(scan_dir, fname))
