# src/descriptor.py
import numpy as np

class ScanContext:
    def __init__(self, num_rings=20, num_sectors=60, max_range=80.0):
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.max_range = max_range


    def compute(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # flip Z: LIDAR is upside down
        z_up = -z 
        
        # filter: 0.2m to 10m above the ground
        mask = (z_up > 0.2) & (z_up < 10.0) 
        x, y, z_up = x[mask], y[mask], z_up[mask]

        # convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2 * np.pi

        # Binning
        # rings
        r_bin = np.floor(r / self.max_range * self.num_rings).astype(int)
        # sectors
        s_bin = np.floor(theta / (2 * np.pi) * self.num_sectors).astype(int)

        # check boundaries
        valid = (r_bin >= 0) & (r_bin < self.num_rings) & \
                (s_bin >= 0) & (s_bin < self.num_sectors)
        
        r_bin, s_bin, z_vals = r_bin[valid], s_bin[valid], z_up[valid]

        desc = np.zeros((self.num_rings, self.num_sectors), dtype=np.float32)
        
        # fill descriptor with maximum height found in each bin
        np.maximum.at(desc, (r_bin, s_bin), z_vals)
        
        return desc
