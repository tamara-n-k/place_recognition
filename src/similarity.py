import numpy as np

def scan_context_nn_distance(sc1, sc2):
    num_sectors = sc1.shape[1]
    # roll sc2 to tensor [shifts, rings, sectors]
    shifts = np.array([np.roll(sc2, s, axis=1) for s in range(num_sectors)])
    
    # compute L1 distance across all shifts at once
    dists = np.mean(np.abs(sc1 - shifts), axis=(1, 2))
    
    return np.min(dists)