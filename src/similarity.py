import numpy as np

def scan_context_nn_distance(sc1, sc2):
    num_sectors = sc1.shape[1]
    min_dist = np.inf
    
    # We must try all shifts to be yaw-invariant
    for shift in range(num_sectors):
        sc2_shifted = np.roll(sc2, shift, axis=1)
        # L1 distance is the SC standard
        dist = np.mean(np.abs(sc1 - sc2_shifted))
        
        if dist < min_dist:
            min_dist = dist
    return min_dist