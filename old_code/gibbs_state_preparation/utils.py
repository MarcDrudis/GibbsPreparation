from scipy.linalg import expm
import numpy as np
import functools
from itertools import product

from qiskit.quantum_info import Statevector,SparsePauliOp,partial_trace,DensityMatrix
from qiskit.circuit import QuantumCircuit



def expected_state(hamiltonian: SparsePauliOp, beta: float):
    state = expm(-beta * hamiltonian.to_matrix())
    state /= np.trace(state)
    return state


def conjugate_pauli(pauli):
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

def printarray(array,rounding=3,func=np.real):
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
    # functools.reduce(lambda a, b: a+b, basis)

    return functools.reduce(lambda a, b: a + b, basis) / np.sqrt(2**num_qubits)

def state_from_ansatz(ansatz: QuantumCircuit, parameters: np.ndarray) -> Statevector:
    """Creates a statevector from an ansatz and parameters."""
    N = ansatz.num_qubits // 2
    return partial_trace(Statevector(ansatz.bind_parameters(parameters)),range(N,2*N) )