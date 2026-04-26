import numpy as np

params_count = 20
params_sets = []

for i in range(params_count):
    lx = np.random.uniform(10.0, 50.0)
    ly = np.random.uniform(10.0, 50.0)
    lz = np.random.uniform(10.0, 40.0)
    gap_size = np.random.uniform(10.0, 20.0)
    pw_error = 10**np.random.uniform(-8, -3) 
    delta_top = np.random.uniform(-1.0, 1.0)
    delta_bot = np.random.uniform(-1.0, 1.0)

    params = {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "gap_size": gap_size,
        "prefactor": 1.0,
        "delta_mid_top": delta_top,
        "delta_mid_bot": delta_bot,
        "pw_error": pw_error,
        "charges": [+1, -1],
    }

    params["positions"] = [
        np.array([
            np.random.uniform(0.1, params["lx"] - 0.1),
            np.random.uniform(0.1, params["ly"] - 0.1),
            np.random.uniform(0.1, params["lz"] - 0.1)
        ]) for _ in range(len(params["charges"]))
    ]

    params_sets.append(params)