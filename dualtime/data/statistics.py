import numpy as np

reps = 5
dirname = "heisencomb/lrel1e1/shots1024/dt5e2_gd100_ws_gd1"
files = [f"{dirname}/{i}.npy" for i in range(1, reps + 1)]

keys = ["energies", "times"]
all_data = {key: [] for key in keys}

# load all data
for filename in files:
    data = np.load(filename, allow_pickle=True).item()
    for key in keys:
        all_data[key].append(data[key])

# compute statistics
statistics = {}
for key in keys:
    statistics[key] = np.mean(all_data[key], axis=0)
    statistics[f"{key}_std"] = np.std(all_data[key], axis=0)

np.save(f"{dirname}/stats.npy", statistics)
