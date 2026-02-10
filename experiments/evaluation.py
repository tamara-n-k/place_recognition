import numpy as np
from src.data_loader import DataLoader
from src.descriptor import ScanContext
from src.similarity import scan_context_nn_distance
import os
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from plot_metrics import plot_evaluation_map, plot_precision_recall_curve


ROOT_DIR = Path(__file__).parent.parent
SCAN_DIR = os.path.join(ROOT_DIR, "data/2012-01-08_vel/2012-01-08/velodyne_sync")
GT_DIR = os.path.join(ROOT_DIR, "data/groundtruth_2012-01-08.csv")

# Parameters
MATCH_THRESHOLD = 1.0  # Scan Context distance threshold
TEMPORAL_WINDOW = 20  # Exclude scans within this many frames
GT_DISTANCE_THRESHOLD = 15.0  # meters for ground truth loop closure
SUBSAMPLING = 10


def scan_timestamp_from_filename(fname):
    return int(fname.replace(".bin", ""))

def get_pose_for_scan(scan_ts, gt_ts, gt_pos):
    idx = np.argmin(np.abs(gt_ts - scan_ts))
    return gt_pos[idx]

def compute_metrics(gt_labels, scores, match_threshold):
    gt_labels = np.array(gt_labels)
    scores = np.array(scores)
    
    # Filter out invalid scores
    valid_mask = np.isfinite(scores)
    gt_labels = gt_labels[valid_mask]
    scores = scores[valid_mask]
    
    if len(gt_labels) == 0 or np.sum(gt_labels) == 0:
        print("No valid ground truth positives!")
        return
    
    # Convert similarity scores to probability-like (lower distance = higher score)
    # Invert so higher is better for sklearn
    prob_scores = 1.0 / (1.0 + scores)
    
    # Compute metrics
    precision, recall, thresholds = precision_recall_curve(gt_labels, prob_scores)
    ap = average_precision_score(gt_labels, prob_scores)
    
    # Compute predictions at threshold
    predictions = (scores < match_threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (gt_labels == 1))
    fp = np.sum((predictions == 1) & (gt_labels == 0))
    fn = np.sum((predictions == 0) & (gt_labels == 1))
    tn = np.sum((predictions == 0) & (gt_labels == 0))
    
    precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_at_threshold = 2 * (precision_at_threshold * recall_at_threshold) / \
                      (precision_at_threshold + recall_at_threshold) \
                      if (precision_at_threshold + recall_at_threshold) > 0 else 0
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total comparisons: {len(gt_labels)}")
    print(f"Ground truth positives: {np.sum(gt_labels)} ({100*np.sum(gt_labels)/len(gt_labels):.1f}%)")
    print(f"Ground truth negatives: {np.sum(1-gt_labels)} ({100*np.sum(1-gt_labels)/len(gt_labels):.1f}%)")
    print(f"\nAverage Precision (AP): {ap:.4f}")
    
    if len(np.unique(gt_labels)) > 1:
        auc = roc_auc_score(gt_labels, prob_scores)
        print(f"AUC-ROC: {auc:.4f}")
    
    print(f"\nAt threshold = {match_threshold}:")
    print(f"  Precision: {precision_at_threshold:.4f}")
    print(f"  Recall: {recall_at_threshold:.4f}")
    print(f"  F1-Score: {f1_at_threshold:.4f}")
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print("="*50 + "\n")
    
    plot_precision_recall_curve(recall, precision, ap)

    return {
        'ap': ap,
        'precision': precision_at_threshold,
        'recall': recall_at_threshold,
        'f1': f1_at_threshold
    }

def compute_descriptors(data, scan):
    print("Phase 1: Computing descriptors...")
    gt_ts, gt_pos = data.load_ground_truth()
    
    descriptors = []
    scan_ids = []
    scan_positions = []
    
    for scan_id, points in data.scan_generator(step=SUBSAMPLING):
        ts = scan_timestamp_from_filename(scan_id)
        pose_xy = get_pose_for_scan(ts, gt_ts, gt_pos)
        
        desc = scan.compute(points)
        
        descriptors.append(desc)
        scan_ids.append(scan_id)
        scan_positions.append(pose_xy)
    
    descriptors = np.array(descriptors)
    scan_positions = np.array(scan_positions)
    
    print(f"Computed {len(descriptors)} descriptors")
    print("\nPhase 2: Matching and evaluation...")
    return descriptors, scan_positions

def run_matching(descriptors, scan_positions):
    # Pre-compute all Ring Keys
    ring_keys = np.array([np.mean(d, axis=1) for d in descriptors])
    
    scores = []
    gt_labels = []
    num_scans = len(descriptors)
    
    # TOP_K determines how many candidates to check
    TOP_K = 50 

    for i in range(num_scans):
        # identify candidates outside temporal window
        candidate_indices = [j for j in range(num_scans) if abs(i - j) >= 50]
        if not candidate_indices: continue
            
        # find Top K loop closure candidates using Ring Keys (L2 Distance)
        query_rk = ring_keys[i]
        cand_rk = ring_keys[candidate_indices]
        
        # Euclidean distance
        rk_dists = np.linalg.norm(cand_rk - query_rk, axis=1)
        
        # Get the indices of the best candidates
        # argsort gives indices relative to cand_rk, so map back to candidate_indices
        best_cand_indices = np.array(candidate_indices)[np.argsort(rk_dists)[:TOP_K]]

        # full Scan Context distance only on the best candidates
        dists = [scan_context_nn_distance(descriptors[i], descriptors[best_cand_indices[j]]) 
                 for j in range(len(best_cand_indices))]
        
        best_dist = min(dists)
        best_j = best_cand_indices[np.argmin(dists)]
        
        # Ground Truth and Results
        gt_dist = np.linalg.norm(scan_positions[i] - scan_positions[best_j])
        gt_labels.append(1 if gt_dist < GT_DISTANCE_THRESHOLD else 0)
        scores.append(best_dist)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{num_scans} scans using Ring Key filter...")

    all_positions = np.array(scan_positions) 
    all_gt_labels = np.array(gt_labels)
    all_scores = np.array(scores)

    plot_evaluation_map(all_positions, all_gt_labels, all_scores, threshold=1.0)
    return np.array(gt_labels), np.array(scores)


if __name__ == "__main__":
    data = DataLoader(SCAN_DIR, GT_DIR)
    scan = ScanContext()
    descriptors, scan_positions = compute_descriptors(data, scan)
    gt_labels, scores = run_matching(descriptors, scan_positions)
    metrics = compute_metrics(gt_labels, scores, MATCH_THRESHOLD)
