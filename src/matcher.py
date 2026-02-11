import numpy as np

class Matcher:
    def __init__(self, top_k=50, temporal_window=50, distance_threshold=15.0):
        self.top_k = top_k
        self.temporal_window = temporal_window      
        self.distance_threshold = distance_threshold  

    def l1_sim(self, sc1, sc2):
        num_sectors = sc1.shape[1]
        # roll sc2 to tensor [shifts, rings, sectors]
        shifts = np.array([np.roll(sc2, s, axis=1) for s in range(num_sectors)])
        
        # compute L1 distance across all shifts at once
        dists = np.mean(np.abs(sc1 - shifts), axis=(1, 2))
        
        return np.min(dists)

    
    def run_matching(self, descriptors, scan_positions):
        # Pre-compute all Ring Keys
        ring_keys = np.array([np.mean(d, axis=1) for d in descriptors])
        
        scores = []
        gt_labels = []
        num_scans = len(descriptors)

        for i in range(num_scans):
            # identify candidates outside temporal window
            candidate_indices = [j for j in range(num_scans) if abs(i - j) >= self.temporal_window]
            if not candidate_indices: continue
                
            # find Top K loop closure candidates using Ring Keys (L2 Distance)
            query_rk = ring_keys[i]
            cand_rk = ring_keys[candidate_indices]
            
            # Euclidean distance
            rk_dists = np.linalg.norm(cand_rk - query_rk, axis=1)
            
            # Get the indices of the best candidates
            # argsort gives indices relative to cand_rk, so map back to candidate_indices
            best_cand_indices = np.array(candidate_indices)[np.argsort(rk_dists)[:self.top_k]]

            # full Scan Context distance only on the best candidates
            dists = [self.l1_sim(descriptors[i], descriptors[best_cand_indices[j]]) 
                    for j in range(len(best_cand_indices))]
            
            best_dist = min(dists)
            best_j = best_cand_indices[np.argmin(dists)]
            
            # Ground Truth and Results
            gt_dist = np.linalg.norm(scan_positions[i] - scan_positions[best_j])
            gt_labels.append(1 if gt_dist < self.distance_threshold else 0)
            scores.append(best_dist)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{num_scans} scans using Ring Key filter...")

        return np.array(gt_labels), np.array(scores)


