import numpy as np
import matplotlib.pyplot as plt

def plot_evaluation_map(positions, gt_labels, scores, threshold=1.0):
    positions = np.array(positions)
    plt.figure(figsize=(10, 10))
    
    # Plot the base trajectory
    plt.plot(positions[:, 0], positions[:, 1], color='gray', alpha=0.3, label='Path')
    
    # Calculate predictions based on threshold
    predictions = (np.array(scores) < threshold).astype(int)
    
    for i in range(len(predictions)):
        x, y = positions[i]
        # True Positive (Green): Correctly found a loop
        if predictions[i] == 1 and gt_labels[i] == 1:
            plt.scatter(x, y, color='green', s=10, alpha=0.5)
        # False Positive (Red): System thought it was a loop, but it wasn't
        elif predictions[i] == 1 and gt_labels[i] == 0:
            plt.scatter(x, y, color='red', s=10, alpha=0.5)
            
    plt.title(f"Loop Closure Results (Threshold={threshold})\nGreen=TP, Red=FP")
    plt.xlabel("X (North) [m]")
    plt.ylabel("Y (East) [m]")
    plt.axis('equal')
    plt.legend(['Path', 'False Positive', 'True Positive'])
    plt.show()

def plot_precision_recall_curve(recall, precision, ap):
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
    