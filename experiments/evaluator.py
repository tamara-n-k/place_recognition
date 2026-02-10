import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score


class Evaluator:
    def __init__(self, matcher, match_threshold, gt_labels, scores):
        self.matcher = matcher
        self.match_threshold = match_threshold
        self.gt_labels = gt_labels
        self.scores = scores

        self.precision = None
        self.recall = None
        self.ap = None

    def compute_metrics(self):
        gt_labels = np.array(self.gt_labels)
        scores = np.array(self.scores)
        
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
        self.precision, self.recall, thresholds = precision_recall_curve(gt_labels, prob_scores)
        self.ap = average_precision_score(gt_labels, prob_scores)
        
        # Compute predictions at threshold
        predictions = (scores < self.match_threshold).astype(int)
        
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
        print(f"\nAverage Precision (AP): {self.ap:.4f}")
        
        if len(np.unique(gt_labels)) > 1:
            auc = roc_auc_score(gt_labels, prob_scores)
            print(f"AUC-ROC: {auc:.4f}")
        
        print(f"\nAt threshold = {self.match_threshold}:")
        print(f"  Precision: {precision_at_threshold:.4f}")
        print(f"  Recall: {recall_at_threshold:.4f}")
        print(f"  F1-Score: {f1_at_threshold:.4f}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print("="*50 + "\n")


            


    def plot_trajectory_map(self, positions):
        positions = np.array(positions)
        plt.figure(figsize=(10, 10))
        
        # Plot the base trajectory
        plt.plot(positions[:, 0], positions[:, 1], color='gray', alpha=0.3, label='Path')
        
        # Calculate predictions based on threshold
        predictions = (np.array(self.scores) < self.match_threshold).astype(int)
        
        for i in range(len(predictions)):
            x, y = positions[i]
            # True Positive (Green): Correctly found a loop
            if predictions[i] == 1 and self.gt_labels[i] == 1:
                plt.scatter(x, y, color='green', s=10, alpha=0.5)
            # False Positive (Red): System thought it was a loop, but it wasn't
            elif predictions[i] == 1 and self.gt_labels[i] == 0:
                plt.scatter(x, y, color='red', s=10, alpha=0.5)
                
        plt.title(f"Loop Closure Results (Threshold={self.match_threshold})\nGreen=TP, Red=FP")
        plt.xlabel("X (North) [m]")
        plt.ylabel("Y (East) [m]")
        plt.axis('equal')
        plt.legend(['Path', 'False Positive', 'True Positive'])
        plt.show()

    def plot_precision_recall_curve(self):
        # Plot PR curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.recall, self.precision, linewidth=2)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(f"Precision-Recall Curve (AP={self.ap:.4f})", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig("precision_recall_curve.png", dpi=300)
        plt.show()
        