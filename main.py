from src.data_loader import DataLoader
from src.descriptor import ScanContext
from src.matcher import Matcher
from experiments.evaluator import Evaluator
from experiments.pipeline import Processor

import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent
SCAN_DIR = os.path.join(ROOT_DIR, "data/2012-01-08_vel/2012-01-08/velodyne_sync")
GT_DIR = os.path.join(ROOT_DIR, "data/groundtruth_2012-01-08.csv")

# Parameters
MATCH_THRESHOLD = 0.8  # Scan Context distance threshold
TEMPORAL_WINDOW = 20  # Exclude scans within this many frames
GT_DISTANCE_THRESHOLD = 15.0  # meters for ground truth loop closure
SUBSAMPLING = 10


data = DataLoader(SCAN_DIR, GT_DIR)
sc = ScanContext(num_rings=40, num_sectors=80, mask_low=0.2, mask_high=10.0)
matcher = Matcher(temporal_window=TEMPORAL_WINDOW, distance_threshold=GT_DISTANCE_THRESHOLD)

# Run Pipeline
processor = Processor(data=data, sc=sc)
descriptors, scan_positions = processor.compute_descriptors(subsampling=SUBSAMPLING)

gt_labels, scores = matcher.run_matching(descriptors, scan_positions)
evaluator = Evaluator(matcher, match_threshold=MATCH_THRESHOLD, gt_labels=gt_labels, scores=scores)

# Print & Plot
metrics = evaluator.compute_metrics()
evaluator.plot_precision_recall_curve()
evaluator.plot_trajectory_map(scan_positions)
evaluator.save_results_to_csv()