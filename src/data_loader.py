import numpy as np
import os
import struct

def load_nclt_scan(bin_path):
    """
    Loads Velodyne .bin
    Returns Nx3 numpy array [m]
    """
    hits = []
    scaling = 0.005  # 5 mm
    offset = -100.0

    with open(bin_path, "rb") as f:
        while True:
            x_bytes = f.read(2)
            if len(x_bytes) < 2:
                break  # EOF

            x = struct.unpack('<H', x_bytes)[0]
            y = struct.unpack('<H', f.read(2))[0]
            z = struct.unpack('<H', f.read(2))[0]
            i = struct.unpack('B', f.read(1))[0]  # intensity
            l = struct.unpack('B', f.read(1))[0]  # laser id (ignore)

            x_m = x * scaling + offset
            y_m = y * scaling + offset
            z_m = z * scaling + offset

            hits.append([x_m, y_m, z_m])

    return np.array(hits)


def scan_generator(scan_dir, step=10):
    files = sorted(os.listdir(scan_dir))
    bin_files = [f for f in files if f.endswith(".bin")]

    for i in range(0, len(bin_files), step):
        fname = bin_files[i]
        yield fname, load_nclt_scan(os.path.join(scan_dir, fname))

