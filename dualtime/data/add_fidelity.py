import numpy as np
from qiskit.quantum_info import Statevector
from models.heisenline import get_model

# reps = 5
# dirname = "heisen3/shots1024_xi1/"
# files = [f"{dirname}/{i}.npy" for i in range(1, reps + 1)]
files = ["heisen3/lr1e3/sv_xi1_nolim.npy"]
baseline = "heisen3/exact.npy"
exact_states = np.load(baseline, allow_pickle=True).item()["states"]

_, circuit, _ = get_model(num_sites=3, reps=3, J=0.25, g=-1)
print(circuit.draw())


def fidelity(reference, parameters):
    sv = Statevector(circuit.bind_parameters(parameters))
    print(sv)
    print(reference)
    return np.abs(np.conj(sv).T.dot(reference)) ** 2


for filename in files:
    data = np.load(filename, allow_pickle=True).item()
    if len(data["parameters"]) != len(exact_states):
        raise ValueError(
            f"Mismatching dimensions {len(data['parameters'])} and {len(exact_states)}!"
        )

    fidelities = [fidelity(ref, theta) for ref, theta in zip(exact_states, data["parameters"])]

    data["fidelities"] = fidelities
    np.save(filename, data)
