import math
import random

import numpy as np


def get_rdm_point(l_x, l_y, l_z):
    return (random.uniform(0, l_x), random.uniform(0, l_y), random.uniform(0, l_z))


def get_rdm_constrained_points(
    l_x, l_y, l_z, point_count=2, min_distance=1.0, max_distance=None
):
    """
    Returns a list of random points within a 3D box.

    Constraints:
    - All points > min_distance from each other.
    - If max_distance is set, each new point must be < max_distance from AT LEAST one existing point.
    """
    points = []
    max_attempts = 1000

    for i in range(point_count):
        found_valid = False
        for _ in range(max_attempts):
            candidate = (
                random.uniform(0, l_x),
                random.uniform(0, l_y),
                random.uniform(0, l_z),
            )

            # Check Minimum Distance (Must be far enough from EVERYONE)
            if any(math.dist(candidate, p) < min_distance for p in points):
                continue

            # Check Maximum Distance (Must be close enough to AT LEAST ONE existing point)
            # We only apply this after the first point is placed.
            if max_distance is not None and i > 0:
                if not any(math.dist(candidate, p) <= max_distance for p in points):
                    continue

            points.append(candidate)
            found_valid = True
            break

        if not found_valid:
            # Could not find a point fitting both constraints within max_attempts
            break

    return points


def get_rdm_constrained_points_np(
    l_x, l_y, l_z, point_count=2, min_distance=1.0, max_distance=None
):
    return [
        np.array(x)
        for x in get_rdm_constrained_points(
            l_x, l_y, l_z, point_count, min_distance, max_distance
        )
    ]
