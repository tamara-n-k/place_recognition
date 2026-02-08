import numpy as np
from src.data_loader import scan_generator, load_ground_truth
from src.descriptor import ScanContext
from src.similarity import scan_context_nn_distance
import os
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent.parent
SCAN_DIR = os.path.join(ROOT_DIR, "data/2012-01-08_vel/2012-01-08/velodyne_sync")
GT_DIR = os.path.join(ROOT_DIR, "data/groundtruth_2012-01-08.csv")
MATCH_THRESHOLD = 0.03
sc = ScanContext()
predictions = []
gt_labels = []
scores = []
descriptors = []
scan_ids = []
scan_positions = []

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

def compute_metrics(gt_labels, scores):
    precision, recall, thresholds = precision_recall_curve(gt_labels, scores)
    ap = average_precision_score(gt_labels, scores)

    print(f"Average Precision (AP): {ap:.4f}")

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()

def run_matching():
    for scan_id, points in scan_generator(SCAN_DIR):
        ts = scan_timestamp_from_filename(scan_id)
        pose_xy = get_pose_for_scan(ts, gt_ts, gt_pos)

        desc = sc.compute(points)

        if len(descriptors) > 0:
            distances = [scan_context_nn_distance(desc, d) for d in descriptors]
            best_dist = min(distances)
            best_idx = distances.index(best_dist)

            gt_dist = np.linalg.norm(pose_xy - scan_positions[best_idx])
            gt_label = int(gt_dist < 5.0)           # ground truth "same place"
            pred = int(best_dist < MATCH_THRESHOLD)  # predicted "same place"

            if pred:
                if gt_label:
                    print(f"{scan_id}: TRUE match (gt dist={gt_dist:.2f} m, sim={best_dist:.3f})")
                else:
                    print(f"{scan_id}: FALSE match (gt dist={gt_dist:.2f} m, sim={best_dist:.3f})")
            else:
                print(f"{scan_id}: No match (best sim={best_dist:.3f})")

            predictions.append(pred)
            gt_labels.append(gt_label)
            scores.append(-best_dist)  # invert for sklearn (higher score == more confident)

        else:
            print(f"{scan_id}: First scan (no previous data)")
            pass

        descriptors.append(desc)
        scan_ids.append(scan_id)
        scan_positions.append(pose_xy)
    return gt_labels, scores

if __name__ == "__main__":
    gt_labels, scores = run_matching()
    compute_metrics(gt_labels, scores)
