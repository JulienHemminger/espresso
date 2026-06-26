import numpy as np

params_count = 20
params_sets = []

for i in range(params_count):
    params = {
        "lx": np.random.uniform(10.0, 50.0),
        "ly": np.random.uniform(10.0, 50.0),
        "lz": np.random.uniform(10.0, 40.0),
        "gap_size": np.random.uniform(10.0, 20.0),
        "prefactor": np.random.uniform(1.0, 2.0),
        "delta_mid_top": np.random.uniform(-1.0, 1.0),
        "delta_mid_bot": np.random.uniform(-1.0, 1.0),
        "pw_error": 1e-8,
        "charges": [+1, -1],
    }

    eps = 1e-2
    params["positions"] = [
        np.array([
            np.random.uniform(eps, params["lx"] - eps),
            np.random.uniform(eps, params["ly"] - eps),
            np.random.uniform(eps, params["lz"] - params["gap_size"] - eps),
        ])
        for _ in range(len(params["charges"]))
    ]

    params_sets.append(params)
