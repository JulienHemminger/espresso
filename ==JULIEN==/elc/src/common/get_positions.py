import random
import math

def get_rdm_point(l_x, l_y, l_z):
    return (
        random.uniform(0, l_x),
        random.uniform(0, l_y),
        random.uniform(0, l_z)
        )

def get_rdm_constrained_points(l_x, l_y, l_z, point_count=2, min_distance=1.0):
    """
    Returns a list of 'point_count' points (float 3-tuple) all in the box from (0, 0, 0) to (l_x, l_y, l_z).

    All points have at least min_distance between each other.
    If the box isnt large enough to fit enough points with the required min_distance inside, return fewer points. 
    """
    points = []
    max_attempts = 1000 
    
    for _ in range(point_count):
        for _ in range(max_attempts):
            # Generate random point
            candidate = get_rdm_point(l_x, l_y, l_z)
            
            # Check distance against all existing points
            if all(math.dist(candidate, p) >= min_distance for p in points):
                points.append(candidate)
                break
        else:
            # If we exhausted attempts without finding a valid point
            break
            
    return points
