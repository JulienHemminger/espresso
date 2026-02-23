import random
import math

def get_rdm_constrained_point_pairs(n, box_size=10.0, min_dist=1.0):
    results = []
    
    for i in range(n):
        # 1. Generate the first position (x, y, z)
        p1 = get_rdm_point(box_size, box_size, box_size)
        
        # 2. Loop until we find a second position far enough away
        while True:
            p2 = get_rdm_point(box_size, box_size, box_size)
            
            # Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
            if math.dist(p1, p2) >= min_dist:
                break
        
        results.append((p1, p2))
        
    return results

def get_rdm_point(l_x, l_y, l_z):
    return (
        random.uniform(0, l_x),
        random.uniform(0, l_y),
        random.uniform(0, l_z)
        )