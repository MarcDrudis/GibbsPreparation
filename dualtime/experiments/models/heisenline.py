import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp


def get_model(num_sites=3, J=1, g=1, reps=1, periodic=False):
    interactions = (
        [("XX", [i, i + 1], J) for i in range(num_sites - 1)]
        + [("YY", [i, i + 1], J) for i in range(num_sites - 1)]
        + [("ZZ", [i, i + 1], J) for i in range(num_sites - 1)]
    )
    if periodic:
        interactions += [(coupling, [num_sites - 1, 0], J) for coupling in ["XX", "YY", "ZZ"]]
    field = [("Z", [i], g) for i in range(num_sites)]
    spo = SparsePauliOp.from_sparse_list(interactions + field, num_sites)

    hamiltonian = PauliSumOp(spo)

    circuit = EfficientSU2(num_sites, reps=reps, entanglement="pairwise")
    # initial_parameters = np.arange(circuit.num_parameters) / circuit.num_parameters
    initial_parameters = np.zeros(circuit.num_parameters)
    for i in range(num_sites):
        initial_parameters[~(i + num_sites)] = np.pi / 2

    return hamiltonian, circuit, initial_parameters
