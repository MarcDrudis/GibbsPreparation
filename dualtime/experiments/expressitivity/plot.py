import numpy as np
import matplotlib.pyplot as plt


def load_reference(n):
    return np.load(f"reference_n{n}.npy", allow_pickle=True).item()


def load_experiment(n, reps):
    return np.load(f"n{n}_r{reps}.npy", allow_pickle=True).item()


def error(reference, experiment, kind="avg"):
    ref = np.asarray(reference["energies"])
    exp = np.asarray(experiment["energies"][1:])
    if kind == "avg":
        return np.mean(np.abs(ref - exp))
    if kind == "max":
        return np.max(np.abs(ref - exp))
    raise NotImplementedError(f"{kind} not supported.")


def landscape(ns, reps, kind="avg"):
    result = np.empty((len(ns), len(reps)))
    for i, n in enumerate(ns):
        reference = load_reference(n)
        for j, rep in enumerate(reps):
            experiment = load_experiment(n, rep)
            result[i, j] = error(reference, experiment, kind=kind)

    return result


ns = [4]
reps = [2, 3, 4]
all_errors = landscape(ns, reps)
plt.imshow(all_errors)
plt.xticks(np.arange(len(reps)), reps)
plt.yticks(np.arange(len(ns)), ns)
plt.colorbar()
plt.show()
