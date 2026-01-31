# src/descriptor.py
import numpy as np

class ScanContext:
    def __init__(self, num_rings=20, num_sectors=60, max_range=80.0):
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.max_range = max_range

        self.ring_size = max_range / num_rings
        self.sector_size = 2 * np.pi / num_sectors

    def compute(self, points):
        desc = np.full(
            (self.num_rings, self.num_sectors),
            -np.inf
        )

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = (theta + 2 * np.pi) % (2 * np.pi)

        mask = r < self.max_range
        r, theta, z = r[mask], theta[mask], z[mask]

        ring_idx = (r / self.ring_size).astype(int)
        sector_idx = (theta / self.sector_size).astype(int)

        ring_idx = np.clip(ring_idx, 0, self.num_rings - 1)
        sector_idx = np.clip(sector_idx, 0, self.num_sectors - 1)

        for i in range(len(r)):
            ri = ring_idx[i]
            si = sector_idx[i]
            desc[ri, si] = max(desc[ri, si], z[i])

        desc[desc == -np.inf] = 0.0

        norm = np.linalg.norm(desc)
        if norm > 0:
            desc /= norm

        return desc
