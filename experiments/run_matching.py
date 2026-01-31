from src.descriptor import ScanContext
from src.data_loader import scan_generator
from src.similarity import scan_context_distance

SCAN_DIR = "data/velodyne"

sc = ScanContext()

descriptors = []
scan_ids = []

for scan_id, points in scan_generator(SCAN_DIR):
    desc = sc.compute(points)
    descriptors.append(desc)
    scan_ids.append(scan_id)

print(f"Loaded {len(descriptors)} scans")

# Example: compare first two
d = scan_context_distance(descriptors[0], descriptors[1])
print("Distance:", d)
