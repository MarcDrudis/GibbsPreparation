#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from itertools import product
import functools
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp, partial_trace
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import expm_multiply, minres, eigsh, expm
from scipy.linalg import ishermitian
from qiskit import QuantumCircuit


def create_hamiltonian_lattice(
    num_sites: int, j_const: float, g_const: float
) -> SparsePauliOp:
    """Creates an Ising Hamiltonian on a lattice."""
    zz_op = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
    x_op = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
    return SparsePauliOp(zz_op) * j_const + SparsePauliOp(x_op) * g_const


def create_constraint_matrix(
    sampled_paulis: dict[str, float], Aq_basis: list, Sm_basis: list
) -> np.ndarray:
    """Creates a constraint matrix from the sampled paulis.

    Args:
        sampled_paulis: A dictionary of sampled paulis and their probabilities.
        Aq_basis: A list of k+1 paulis for the q coordinate.
        Sm_basis: A list of k paulis for the m coordinate.
    """
    data = []
    row = []
    col = []
    for i, Aq_label in enumerate(Aq_basis):
        Aq_Pauli = Pauli(Aq_label)
        for j, Sm_label in enumerate(Sm_basis):
            Sm_Pauli = Pauli(Sm_label)
            if Aq_Pauli.anticommutes(Sm_Pauli):
                operator = 1j * Aq_Pauli @ Sm_Pauli
                phase = 1j**operator.phase
                value = phase * sampled_paulis[(operator * phase).to_label()]
                if np.abs(value) != 0:
                    row.append(i)
                    col.append(j)
                    data.append(value)
            elif not Aq_Pauli.commutes(Sm_Pauli):
                raise ValueError("Paulis do not commute or anticommute.")

    return csr_matrix((data, (row, col)), shape=(len(Aq_basis), len(Sm_basis)))


def create_klocal_pauli_basis(num_qubits: int, k: int):
    """Creates a k-local pauli basis."""
    if k == 0:
        return set()
    else:
        blocks = {"".join(i) for i in product(["I", "X", "Y", "Z"], repeat=k)}
        pauli_basis = {
            "I" * shift + block + "I" * (num_qubits - k - shift)
            for block, shift in product(blocks, range(num_qubits + 1 - k))
        }
        pauli_basis.remove("I" * num_qubits)
        return pauli_basis


def sample_pauli_basis(
    state: Statevector | QuantumCircuit,
    pauli_basis: list,
    num_samples: int,
    noise: float = 0.0,
) -> dict[str, float]:
    """Creates a dictionary of sampled paulis and their probabilities from a given
    statevector and pauli basis to sample from."""
    sampled_probs = {}
    for pauli in pauli_basis:
        pauli_op = Pauli(pauli + "I" * len(pauli))
        expectation_value = state.expectation_value(pauli_op)
        expectation_value += np.random.normal(0, noise)
        sampled_probs[
            pauli
        ] = expectation_value  # if np.abs(expectation_value) >1e-7 else 0
    return sampled_probs


# I should be able to have identities between the terms in the blocks and I should be able to handle
# #far away blocks when computing the expetnace of commutators


def identity_purification(num_qubits: int) -> Statevector:
    """Creates a statevector purification of the identity."""
    basis = [
        Statevector.from_label(2 * "".join(element))
        for element in product(["0", "1"], repeat=num_qubits)
    ]
    # functools.reduce(lambda a, b: a+b, basis)

    return functools.reduce(lambda a, b: a + b, basis) / np.sqrt(2**num_qubits)


def simple_purify_hamiltonian(
    hamiltonian: SparsePauliOp, noise: float = 0
) -> Statevector:
    """Creates a statevector purification of a hamiltonian."""
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


def reconstruct_hamiltonian(pauli_basis: list, coefficients: np.array) -> SparsePauliOp:
    """Reconstructs a hamiltonian from a pauli basis and a list of coefficients."""
    reconstructed_hamiltonian = [
        (label, weight)
        for label, weight in zip(pauli_basis, coefficients)
        if np.abs(weight) > 0
    ]

    return SparsePauliOp.from_list(reconstructed_hamiltonian).simplify()


def hamiltonian_to_vector(pauli_basis: list, hamiltonian: SparsePauliOp) -> np.array:
    """Creates a vector representation of a hamiltonian for a given PauliBasis."""
    vector = np.zeros(len(pauli_basis), dtype=complex)
    for label, weight in hamiltonian.label_iter():
        vector[pauli_basis.index(label)] = weight
    return vector
