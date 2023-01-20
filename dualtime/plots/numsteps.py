import numpy as np
import matplotlib.pyplot as plt

mode = "heisen"
if mode == "heisen":
    settings = [
        # ("data/heisen3/varqite.npy", "VarQITE", "crimson", "-"),
        # ("data/heisen3/sv_xi1.npy", "dual, statevector", "royalblue", "-"),
        # ("data/heisen3/sv_xi1_ws.npy", "dual, statevector", "royalblue", "-"),
        # ("data/heisen3/lr1e3/sv_xi1_nolim.npy", "dual, statevector", "royalblue", "-"),
        ("data/heisen3/sv_xi1.npy", "SV, $\eta=0.01$", "royalblue", "-"),
        ("data/heisen3/lr1e3/sv_xi1.npy", "dual, $\eta=0.001$", "seagreen", "--"),
        ("data/heisen3/lr1e3/sv_xi1_nolim.npy", "dual, $\eta=0.001$, nolim", "crimson", "-."),
    ]


def plot(filename, label, color, ls):
    data = np.load(filename, allow_pickle=True).item()
    plt.plot(data["times"][:-1], data["num_steps"], color=color, label=label, ls=ls, marker="o")
    if "num_steps_std" in data.keys():
        plt.fill_between(
            data["times"][:-1],
            data["num_steps"] - data["num_steps_std"],
            data["num_steps"] + data["num_steps_std"],
            alpha=0.2,
            color=color,
        )


plt.figure(figsize=(4, 3))
for setting in settings:
    plot(*setting)

plt.title(f"Ising {mode}")
plt.xlabel("time $t$")
plt.ylabel("optimization steps")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
