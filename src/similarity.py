import numpy as np

def scan_context_distance(desc1, desc2):
    min_dist = np.inf
    num_sectors = desc1.shape[1]

    for shift in range(num_sectors):
        shifted = np.roll(desc2, shift, axis=1)
        dist = np.linalg.norm(desc1 - shifted)
        min_dist = min(min_dist, dist)

    return min_dist
