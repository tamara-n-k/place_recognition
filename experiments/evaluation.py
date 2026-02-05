import numpy as np
from src.data_loader import scan_generator
from src.descriptor import ScanContext
from src.similarity import scan_context_nn_distance
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
print(ROOT_DIR)
SCAN_DIR = os.path.join(ROOT_DIR, "data/2012-01-08_vel/2012-01-08/velodyne_sync")
GT_DIR = os.path.join(ROOT_DIR, "data/groundtruth_2012-01-08.csv")
MATCH_THRESHOLD = 0.03
sc = ScanContext()

def load_ground_truth(gt_csv):
    gt = np.loadtxt(gt_csv, delimiter=",")
    timestamps = gt[:, 0]
    positions = gt[:, 1:3]  # x, y
    return timestamps, positions

def scan_timestamp_from_filename(fname):
    return int(fname.replace(".bin", ""))

def get_pose_for_scan(scan_ts, gt_ts, gt_pos):
    idx = np.argmin(np.abs(gt_ts - scan_ts))
    return gt_pos[idx]

def find_best_match(i, descriptors):
    best_j = None
    best_dist = np.inf

    for j in range(len(descriptors)):
        if abs(i - j) < 30:
            continue  # ignore temporally close scans

        d = scan_context_nn_distance(descriptors[i], descriptors[j])
        if d < best_dist:
            best_dist = d
            best_j = j

    return best_j, best_dist

gt_ts, gt_pos = load_ground_truth(GT_DIR)

descriptors = []
scan_ids = []
scan_positions = []

for scan_id, points in scan_generator(SCAN_DIR):
    ts = scan_timestamp_from_filename(scan_id)
    pose_xy = get_pose_for_scan(ts, gt_ts, gt_pos)

    desc = sc.compute(points)

    if len(descriptors) > 0:
        distances = [scan_context_nn_distance(desc, d) for d in descriptors]
        best_dist = min(distances)
        best_idx = distances.index(best_dist)

        gt_dist = np.linalg.norm(pose_xy - scan_positions[best_idx])

        if best_dist < MATCH_THRESHOLD:
            if gt_dist < 1.0:
                print(f"{scan_id}: TRUE match ({gt_dist:.1f} m)")
            else:
                print(f"{scan_id}: FALSE match ({gt_dist:.1f} m)")
        else:
            print(f"{scan_id}: No match")
    else:
        print(f"{scan_id}: First scan")

    descriptors.append(desc)
    scan_ids.append(scan_id)
    scan_positions.append(pose_xy)