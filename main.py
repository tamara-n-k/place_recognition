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
MATCH_THRESHOLD = 1.0  # Scan Context distance threshold
TEMPORAL_WINDOW = 20  # Exclude scans within this many frames
GT_DISTANCE_THRESHOLD = 15.0  # meters for ground truth loop closure
SUBSAMPLING = 10


data = DataLoader(SCAN_DIR, GT_DIR)
sc = ScanContext()
matcher = Matcher(distance_threshold=GT_DISTANCE_THRESHOLD)
evaluator = Evaluator(matcher, match_threshold=MATCH_THRESHOLD)

# Run Pipeline
processor = Processor(data=data, sc=sc)
descriptors, scan_positions = processor.compute_descriptors(subsampling=SUBSAMPLING)

gt_labels, scores = matcher.run_matching(descriptors, scan_positions)

# Print & Plot
metrics = evaluator.compute_metrics(gt_labels, scores)
# evaluator.plot_precision_recall_curve(evaluator.recall, evaluator.precision, evaluator.ap)
# evaluator.plot_trajectory_map()