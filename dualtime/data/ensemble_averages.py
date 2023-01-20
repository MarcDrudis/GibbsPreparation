import numpy as np

reps = 10
dirname = "heisen2/qmetts20_stop"
files = [f"{dirname}/{i + 1}.npy" for i in range(1, reps + 1)]

keys = ["averages"]
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

print(statistics)
# np.save(f"{dirname}/stats.npy", statistics)
