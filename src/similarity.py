import numpy as np

def scan_context_nn_distance(desc1, desc2):
    min_dist = np.inf
    num_sectors = desc1.shape[1]

    for shift in range(num_sectors):
        shifted = np.roll(desc2, shift, axis=1)
        dist = nn_column_distance(desc1, shifted)
        min_dist = min(min_dist, dist)

    return min_dist


def nn_column_distance(desc1, desc2):
    """
    Nearest-neighbor column distance between two descriptors
    """
    num_sectors = desc1.shape[1]
    total_dist = 0.0

    for j in range(num_sectors):
        col1 = desc1[:, j]

        dists = np.linalg.norm(
            desc2 - col1[:, None],
            axis=0
        )

        total_dist += np.min(dists)

    return total_dist / num_sectors
