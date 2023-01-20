import numpy as np

# reps = 5
# dirname = "heisen3/shots1024_xi1/"
# files = [f"{dirname}/{i}.npy" for i in range(1, reps + 1)]
files = ["heisen3/lr1e3/sv_xi1_nolim.npy"]

for filename in files:
    data = np.load(filename, allow_pickle=True).item()
    num_steps = [len(losses_i) for losses_i in data["losses"]]
    data["num_steps"] = num_steps
    np.save(filename, data)
