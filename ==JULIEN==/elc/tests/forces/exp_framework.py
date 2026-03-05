def run_test(lx, ly, lz, gap_size, pw_error=1e-6, tolerance=1e-6, charges=[]):
    
    particle_count = len(charges)
    # generate positions