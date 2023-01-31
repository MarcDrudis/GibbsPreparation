from __future__ import annotations
from scipy.linalg import expm
import numpy as np
import functools
from itertools import product
from qiskit.quantum_info import (
    Statevector,
    SparsePauliOp,
    partial_trace,
    DensityMatrix,
    Pauli,
)
from qiskit.circuit import QuantumCircuit
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import logm
from gibbs.learning.hamiltonian_learning import HamiltonianLearning
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis


def expected_state(hamiltonian: SparsePauliOp, beta: float):
    """Computes the mixed Gibbs state of a given hamiltonian at a given temperature."""
    state = expm(-beta * hamiltonian.to_matrix())
    state /= np.trace(state)
    return DensityMatrix(state)


def conjugate_pauli(pauli: str) -> str:
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


def printarray(array, rounding=3, func=np.real, scientific=False):
    """Prints a numpy array with a given rounding and function
    to deal with complex values."""
    if scientific:
        np.set_printoptions(suppress=True)
        print(array)
        np.set_printoptions(suppress=False)
    if func == None:
        print(np.round(array, rounding))
    else:
        print(np.round(func(array), rounding))


def lattice_hamiltonian(
    num_sites: int,
    j_const: float,
    g_const: float,
    one_local: list[str],
    two_local: list[str],
):
    for o in one_local:
        if len(o) != 1:
            raise AssertionError("Lenght of one local fields is not correct")
    for t in two_local:
        if len(t) != 2:
            raise AssertionError("Lenght of two local fields is not correct")

    two_ops = []
    for t in two_local:
        two_ops += [
            "I" * i + t + "I" * (num_sites - i - 2) for i in range(num_sites - 1)
        ]

    one_ops = []
    for o in one_local:
        one_ops += ["I" * i + o + "I" * (num_sites - i - 1) for i in range(num_sites)]

    return SparsePauliOp(two_ops) * j_const + SparsePauliOp(one_ops) * g_const


def create_hamiltonian_lattice(
    num_sites: int,
    j_const: float,
    g_const: float,
) -> SparsePauliOp:
    """Creates an Ising Hamiltonian on a lattice."""
    zz_op = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
    x_op = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
    return SparsePauliOp(zz_op) * j_const + SparsePauliOp(x_op) * g_const


def create_heisenberg(
    num_sites: int, j_const: float, g_const: float, circular: bool = False
) -> SparsePauliOp:
    """Creates an Heisenberg Hamiltonian on a lattice."""
    xx_op = ["I" * i + "XX" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
    yy_op = ["I" * i + "YY" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
    zz_op = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]

    circ_op = (
        ["X" + "I" * (num_sites - 2) + "X"]
        + ["Y" + "I" * (num_sites - 2) + "Y"]
        + ["Z" + "I" * (num_sites - 2) + "Z"]
        if circular
        else []
    )

    z_op = ["I" * i + "Z" + "I" * (num_sites - i - 1) for i in range(num_sites)]

    return (
        SparsePauliOp(xx_op + yy_op + zz_op + circ_op) * j_const
        + SparsePauliOp(z_op) * g_const
    )


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


def number_of_elements(k, n):  # (n-k+1) * (3/4)**2 * 4**k
    if k == 1:
        return 3 * n

    # The inner block of operators will be formed by {I,X,Y,Z}^{k-2}
    inner_block = 4 ** (k - 2)
    # The blocks of this border need to be formed by {X,Y,Z}^2. If we have an identity in the border it would be a k-1 local term.
    outer_block = 3**2
    # Finally this block of operators is free to move around the lattice.
    shifting = n - k + 1  # This would just be n for periodic boundary conditions
    return inner_block * outer_block * shifting


def classical_learn_hamiltonian(
    state: QuantumCircuit | DensityMatrix, klocality: int
) -> np.ndarray:
    if isinstance(state, QuantumCircuit):
        num_qubits = state.num_qubits // 2
        mixed_state = partial_trace(Statevector(state), range(num_qubits))
    elif isinstance(state, DensityMatrix):
        num_qubits = int(np.log2(state.shape[0]))
        mixed_state = state.data
    else:
        print("Not supported type")

    learning_basis = KLocalPauliBasis(klocality, num_qubits)
    hamiltonian_cl_rec = -logm(mixed_state.data)
    hamiltonian_cl_rec = hamiltonian_cl_rec  # - np.eye(hamiltonian_cl_rec.shape[0])*np.trace(hamiltonian_cl_rec)/hamiltonian_cl_rec.shape[0]
    recov_vec = [
        np.trace(hamiltonian_cl_rec @ Pauli(p).to_matrix())
        for p in learning_basis._paulis_list
    ]
    recov_vec = np.array([v / hamiltonian_cl_rec.shape[0] for v in recov_vec])
    return recov_vec
