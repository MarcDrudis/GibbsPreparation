from scipy.linalg import expm
import numpy as np
import functools
from itertools import product

from qiskit.quantum_info import Statevector, SparsePauliOp, partial_trace, DensityMatrix
from qiskit.circuit import QuantumCircuit
from scipy.sparse.linalg import expm_multiply


def expected_state(hamiltonian: SparsePauliOp, beta: float):
    """Computes the mixed Gibbs state of a given hamiltonian at a given temperature."""
    state = expm(-beta * hamiltonian.to_matrix())
    state /= np.trace(state)
    return state


def conjugate_pauli(pauli: str)->str:
    """For a given pauli string returns the pauli string such that the product of both
    will yield a non-zero imaginary value when evaluated at (|00>+|11>)^n.
    """

    d = {"X": "Y", "Y": "Z", "Z": "X"}
    dd = {"X": "Z", "Y": "X", "Z": "Y"}
    for i, s in enumerate(pauli):
        if s != "I":
            return (
                "I" * i
                + dd[s]
                + "I" * (len(pauli) - i - 1)
                + pauli[:i]
                + d[s]
                + pauli[i + 1 :]
            )
    return "I" * 2 * len(pauli)


def printarray(array, rounding=3, func=np.real):
    """Prints a numpy array with a given rounding and function
    to deal with complex values."""
    if func == None:
        print(np.round(array,rounding))
    else:
        print(np.round(func(array), rounding))


def create_hamiltonian_lattice(
    num_sites: int, j_const: float, g_const: float
) -> SparsePauliOp:
    """Creates an Ising Hamiltonian on a lattice."""
    zz_op = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
    x_op = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
    return SparsePauliOp(zz_op) * j_const + SparsePauliOp(x_op) * g_const


def identity_purification(num_qubits: int) -> Statevector:
    """Creates a statevector purification of the identity."""
    basis = [
        Statevector.from_label(2 * "".join(element))
        for element in product(["0", "1"], repeat=num_qubits)
    ]
    return functools.reduce(lambda a, b: a + b, basis) / np.sqrt(2**num_qubits)


def state_from_ansatz(ansatz: QuantumCircuit, parameters: np.ndarray) -> Statevector:
    """Creates a statevector from an ansatz and parameters."""
    N = ansatz.num_qubits // 2
    return partial_trace(
        Statevector(ansatz.bind_parameters(parameters)), range(N, 2 * N)
    )


def simple_purify_hamiltonian(
    hamiltonian: SparsePauliOp, noise: float = 0
) -> Statevector:
    """Creates a statevector purification of the thermal state of the hamiltonian."""
    extended_hamiltonian = hamiltonian ^ ("I" * hamiltonian.num_qubits)
    sparse_hamiltonian = extended_hamiltonian.to_matrix(sparse=True)
    id_pur = identity_purification(hamiltonian.num_qubits)
    state = expm_multiply(
        -sparse_hamiltonian / 2,
        id_pur.data,
    )
    state = state / np.linalg.norm(state)
    state += np.random.normal(0, noise, len(state)) * np.exp(
        1j * np.random.uniform(0, 2 * np.pi, len(state))
    )
    state = state / np.linalg.norm(state)
    return Statevector(state)