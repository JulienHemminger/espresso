import numpy as np

def are_sets_equal(set_a, set_b, tol=1e-6):
    if len(set_a) != len(set_b):
        return False
    
    # 1. Sort the individual elements within the lists
    # We use lexsort or simply convert to sorted lists to handle the 'unordered' nature
    sort_a = sorted(set_a, key=lambda x: tuple(x))
    sort_b = sorted(set_b, key=lambda x: tuple(x))
    
    # 2. Use np.allclose to compare the sorted structures
    return np.allclose(sort_a, sort_b, atol=tol)