import numpy as np

class Processor:
    def __init__(self, data, sc):
        self.data = data
        self.sc = sc

    def scan_timestamp_from_filename(self, fname):
        return int(fname.replace(".bin", ""))

    def get_pose_for_scan(self, scan_ts, gt_ts, gt_pos):
        idx = np.argmin(np.abs(gt_ts - scan_ts))
        return gt_pos[idx]
    
    def compute_descriptors(self, subsampling):
            print("Computing descriptors...")
            gt_ts, gt_pos = self.data.load_ground_truth()
            
            descriptors = []
            scan_ids = []
            scan_positions = []
            
            for scan_id, points in self.data.scan_generator(step=subsampling):
                ts = self.scan_timestamp_from_filename(scan_id)
                pose_xy = self.get_pose_for_scan(ts, gt_ts, gt_pos)
                
                desc = self.sc.compute(points)
                
                descriptors.append(desc)
                scan_ids.append(scan_id)
                scan_positions.append(pose_xy)
            
            descriptors = np.array(descriptors)
            scan_positions = np.array(scan_positions)
            
            print(f"Computed {len(descriptors)} descriptors")
            return descriptors, scan_positions
