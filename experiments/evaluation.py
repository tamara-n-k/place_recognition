import numpy as np
from src.data_loader import scan_generator, load_ground_truth
from src.descriptor import ScanContext
from src.similarity import scan_context_nn_distance
import os
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent.parent
SCAN_DIR = os.path.join(ROOT_DIR, "data/2012-01-08_vel/2012-01-08/velodyne_sync")
GT_DIR = os.path.join(ROOT_DIR, "data/groundtruth_2012-01-08.csv")

# Parameters
MATCH_THRESHOLD = 0.5  # Scan Context distance threshold (typically 0.3-0.7)
TEMPORAL_WINDOW = 30   # Exclude scans within this many frames
GT_DISTANCE_THRESHOLD = 5.0  # meters for ground truth loop closure

sc = ScanContext()

def scan_timestamp_from_filename(fname):
    return int(fname.replace(".bin", ""))

def get_pose_for_scan(scan_ts, gt_ts, gt_pos):
    idx = np.argmin(np.abs(gt_ts - scan_ts))
    return gt_pos[idx]

def compute_metrics(gt_labels, scores, match_threshold):
    """Compute comprehensive evaluation metrics"""
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
    
    # Plot PR curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve (AP={ap:.4f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=300)
    plt.show()
    
    return {
        'ap': ap,
        'precision': precision_at_threshold,
        'recall': recall_at_threshold,
        'f1': f1_at_threshold
    }

def compute_descriptors():
    print("Phase 1: Computing descriptors...")
    gt_ts, gt_pos = load_ground_truth(GT_DIR)
    
    descriptors = []
    scan_ids = []
    scan_positions = []
    
    for scan_id, points in scan_generator(SCAN_DIR):
        ts = scan_timestamp_from_filename(scan_id)
        pose_xy = get_pose_for_scan(ts, gt_ts, gt_pos)
        
        desc = sc.compute(points)
        
        descriptors.append(desc)
        scan_ids.append(scan_id)
        scan_positions.append(pose_xy)
    
    descriptors = np.array(descriptors)
    scan_positions = np.array(scan_positions)
    
    print(f"Computed {len(descriptors)} descriptors")
    print("\nPhase 2: Matching and evaluation...")
    return descriptors, scan_positions

def run_matching(descriptors, scan_positions):
    predictions = []
    gt_labels = []
    scores = []
    
    for i in range(len(descriptors)):
        # Find valid candidates (outside temporal window)
        candidate_indices = [j for j in range(len(descriptors)) 
                           if abs(i - j) >= TEMPORAL_WINDOW]
        
        if not candidate_indices:
            continue
        
        # Compute distances using Scan Context metric
        dists = np.array([scan_context_nn_distance(descriptors[i], descriptors[j]) 
                         for j in candidate_indices])
        
        # Find best match
        min_idx = np.argmin(dists)
        best_dist = dists[min_idx]
        best_j = candidate_indices[min_idx]
        
        # Ground truth: spatial distance
        gt_dist = np.linalg.norm(scan_positions[i] - scan_positions[best_j])
        gt_label = int(gt_dist < GT_DISTANCE_THRESHOLD)
        pred = int(best_dist < MATCH_THRESHOLD)
        
        predictions.append(pred)
        gt_labels.append(gt_label)
        scores.append(best_dist)  # Keep as distance (lower is better)
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(descriptors)} scans")
    
    return np.array(gt_labels), np.array(scores), np.array(predictions)

if __name__ == "__main__":
    descriptors, scan_positions = compute_descriptors()
    gt_labels, scores, predictions = run_matching(descriptors, scan_positions)
    metrics = compute_metrics(gt_labels, scores, MATCH_THRESHOLD)
