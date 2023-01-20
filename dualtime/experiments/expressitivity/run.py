import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp

from varqite import ConvQite


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


ns = [2, 4, 6, 8]
reps = [2, 3, 4, 5, 6, 7, 8]
first = True

for n in ns:
    refname = f"reference_n{n}.npy"
    for rep in reps:
        print(f"Evaluating n = {n}, reps = {rep}...")
        hamiltonian, circuit, initial_parameters = get_model(n, reps=rep, J=0.25, g=-1)
        qite = ConvQite(circuit, initial_parameters)

        if first:
            hamiltonian_matrix, ref = qite.classical_reference(hamiltonian)
            np.save(refname, ref)
            first = False

        res = qite.quantum_evolve(hamiltonian_matrix, ref)
        np.save(f"n{n}_r{reps}.npy", res)
