import numpy as np
import matplotlib.pyplot as plt

mode = "ising3"
index = 0

if mode == "ising3":
    settings = [
        ("ising.npy", r"dual", "royalblue", "-"),
    ]


def get_observables(observables):
    return np.array([obs[index] for obs in observables])


def plot(filename, label, color, ls):
    data = np.load(filename, allow_pickle=True).item()
    plt.plot(data["times"], data["observables"], color=color, label=label, ls=ls)
    if "energies_std" in data.keys():
        plt.fill_between(
            data["times"],
            data["energies"] - data["energies_std"],
            data["energies"] + data["energies_std"],
            alpha=0.2,
            color=color,
        )


plt.figure(figsize=(4, 3))
for setting in settings:
    plot(*setting)

# plt.xlim(0, 1)
plt.title(f"12q Heisenberg model on a line")
plt.xlabel("time $t$")
plt.ylabel("energy $E$")
# plt.legend(loc="best")
plt.legend(bbox_to_anchor=(1, 1))
# plt.tight_layout()
plt.show()
