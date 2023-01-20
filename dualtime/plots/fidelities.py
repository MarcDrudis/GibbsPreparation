import numpy as np
import matplotlib.pyplot as plt

mode = "heisen"
if mode == "heisen":
    settings = [
        ("data/heisen3/varqite.npy", "VarQITE", "crimson", "-"),
        # ("data/heisen3/sv_xi1.npy", "dual, statevector", "royalblue", "-"),
        # ("data/heisen3/sv_xi1_ws.npy", "dual, statevector", "royalblue", "-"),
        # ("data/heisen3/lr1e3/sv_xi1_nolim.npy", "dual, statevector", "royalblue", "-"),
        ("data/heisen3/sv_xi1.npy", "dual, statevector", "royalblue", "-"),
        ("data/heisen3/lr1e3/sv_xi1.npy", "dual, statevector", "royalblue", "--"),
        ("data/heisen3/lr1e3/sv_xi1_nolim.npy", "dual, statevector", "royalblue", "-."),
    ]


def plot(filename, label, color, ls):
    data = np.load(filename, allow_pickle=True).item()
    plt.plot(data["times"], data["fidelities"], color=color, label=label, ls=ls)
    if "fidelities_std" in data.keys():
        plt.fill_between(
            data["times"],
            data["fidelities"] - data["fidelities_std"],
            data["fidelities"] + data["fidelities_std"],
            alpha=0.2,
            color=color,
        )


plt.figure(figsize=(4, 3))
for setting in settings:
    plot(*setting)

plt.title(f"Ising {mode}")
plt.xlabel("time $t$")
plt.ylabel("energy $E$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
