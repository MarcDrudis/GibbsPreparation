import numpy as np
import matplotlib.pyplot as plt

mode = "heisen2"
if mode == "triangle":
    settings = [
        ("data/triangle/exact.npy", "exact", "k", "-"),
        ("data/triangle/varqite.npy", "VarQITE", "crimson", "-"),
        ("data/triangle/shots1024_xi1/stats.npy", r"dual, statevector, $\xi=1$", "royalblue", "-"),
        ("data/triangle/shots1024_xi1/stats.npy", r"dual, 1024 shots, $\xi=1$", "royalblue", "--"),
        ("data/triangle/shots/1.npy", "1", "crimson", "--")
        # ("data/triangle/shots1024_xi1_ws/stats.npy", r"warmss, $\xi=1$", "seagreen"),
    ]
elif mode == "comb":
    settings = [
        ("data/comb/exact_sine.npy", r"exact", "k"),
        # ("data/comb/exact_sine_offset.npy", r"exact", "k"),
        ("data/comb/varqite_sine.npy", "VarQITE", "crimson"),
        # ("data/comb/sv_xi1_ws.npy", r"SV, ws, $\xi=1$", "seagreen"),
    ]
elif mode == "hex":
    settings = [
        ("data/hex/exact_sine_p1.npy", r"exact", "k"),
        ("data/hex/varqite_sine_r3.npy", "VarQITE", "crimson"),
    ]
elif mode == "heisen":
    settings = [
        ("data/heisen3/exact.npy", r"exact", "k", "-"),
        ("data/heisen3/varqite.npy", "VarQITE", "crimson", "-"),
        # ("data/heisen3/lr1e3/sv_xi1.npy", "dual, statevector, GD", "royalblue", "-"),
        # (
        #     "data/heisen3/lr1e3/spsa_it100_b1/sv_xi1.npy",
        #     "dual, statevector, SPSA",
        #     "royalblue",
        #     "--",
        # ),
        ("data/heisen3/lr1e2/shots_xi1/stats.npy", "dual, 1024 shots", "seagreen", "-"),
        # (
        #     "data/heisen3/lr1e3/spsa_it100_b1/shots_xi1/1.npy",
        #     "dual, statevector, SPSA",
        #     "seagreen",
        #     "--",
        # ),
        # ("data/heisen3/lrel1e1/xi0/shots1024/spsa_it100_b1_0s/1.npy", "trial", "tab:orange", "-"),
        # learning rate to small: no convergence!
        # ("data/heisen3/lr1e3/shots_xi1/stats.npy", "dual, 1024 shots", "royalblue", "--"),
        # ("data/heisen3/sv_xi1_ws.npy", "dual, statevector", "royalblue", "-"),
        # ("data/heisen3/lr1e3/sv_xi1_nolim.npy", "dual, statevector", "royalblue", "-"),
        # ("data/heisen3/lr1e3/sv_xi1.npy", "dual, statevector", "royalblue", "-"),
    ]
elif mode == "heisen2":
    settings = [
        ("data/heisen2/00_exact.npy", r"00", "royalblue", "-"),
        ("data/heisen2/lrel1e1/sv/00_gd100.npy", None, "royalblue", "--"),
        ("data/heisen2/10_exact.npy", r"10", "seagreen", "-"),
        ("data/heisen2/lrel1e1/sv/10_gd100.npy", None, "seagreen", "--"),
        ("data/heisen2/01_exact.npy", r"01", "crimson", "-"),
        ("data/heisen2/lrel1e1/sv/01_gd100.npy", None, "crimson", "--"),
        ("data/heisen2/11_exact.npy", r"11", "goldenrod", "-"),
        ("data/heisen2/lrel1e1/sv/11_gd100.npy", None, "goldenrod", "--"),
    ]
elif mode == "heisen2X":
    settings = [
        ("data/heisen2/++_exact.npy", r"exact", "royalblue", "-"),
        ("data/heisen2/lrel1e1/sv/++_gd100.npy", "dual", "royalblue", "--"),
        ("data/heisen2/-+_exact.npy", r"exact", "seagreen", "-"),
        ("data/heisen2/lrel1e1/sv/-+_gd100.npy", "dual", "seagreen", "--"),
        ("data/heisen2/+-_exact.npy", r"exact", "crimson", "-"),
        ("data/heisen2/lrel1e1/sv/+-_gd100.npy", "dual", "crimson", "--"),
        ("data/heisen2/--_exact.npy", r"exact", "goldenrod", "-"),
        ("data/heisen2/lrel1e1/sv/--_gd100.npy", "dual", "goldenrod", "--"),
    ]
elif mode == "heisen8":
    settings = [
        ("data/heisen8/exact.npy", r"exact", "k", "-"),
        # ("data/heisen3/varqite.npy", "VarQITE", "crimson", "-"),
        ("data/heisen8/lrel1e1/xi0/sv/spsa_it100_b1_0s/1.npy", "dual, SV, SPSA", "seagreen", "--"),
        # (
        #     "data/heisen8/lrel1e1/xi0/shots1024/spsa_it100_b1_0s/1.npy",
        #     "dual, 1024 shots, SPSA",
        #     "royalblue",
        #     "--",
        # ),
        (
            "data/heisen8/lrel1e1/xi0/sv/spsa_it100_b1_c1e1_ws/1.npy",
            "dual, SV, SPSA",
            "royalblue",
            "--",
        ),
    ]
elif mode == "heisen12":
    settings = [
        ("data/heisencomb/exact.npy", r"exact", "k", "-"),
        # ("data/heisen3/varqite.npy", "VarQITE", "crimson", "-"),
        ("data/heisencomb/lrel1e1/sv/gd.npy", "SV, 0-start (100)", "royalblue", "-"),
        (
            "data/heisencomb/lrel1e1/shots1024/gd100_ws_gd1/stats.npy",
            "1024, warmstart (100-1)",
            "seagreen",
            ":",
        ),
        (
            "data/heisencomb/lrel1e1/shots1024/dt5e2_gd100_ws_gd1/stats.npy",
            "1024, warmstart (100-1), dt=0.05",
            "crimson",
            ":",
        ),
        (
            "data/heisencomb/lrel1e1/shots1024/gd100_ws_gd10/stats.npy",
            "1024, warmstart (100-10)",
            "seagreen",
            "-",
        ),
        # ("data/heisencomb/1.npy", "1024 shots, warmstart (100-10)", "seagreen", "--"),
        (
            "data/heisencomb/lrel1e1/shots1024/gd100_ws_nostop/stats.npy",
            "1024 shots, warmstart (100-100)",
            "seagreen",
            "--",
        ),
    ]


def plot(filename, label, color, ls):
    data = np.load(filename, allow_pickle=True).item()
    plt.plot(data["times"], data["energies"], color=color, label=label, ls=ls)
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

plt.xlim(0, 1)
plt.title(f"12q Heisenberg model on a line")
plt.xlabel("time $t$")
plt.ylabel("energy $E$")
# plt.legend(loc="best")
plt.legend(bbox_to_anchor=(1, 1))
# plt.tight_layout()
plt.show()
