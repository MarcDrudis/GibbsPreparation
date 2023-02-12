from __future__ import annotations

import functools
from itertools import product

import numpy as np
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.primitives import Estimator
from qiskit.quantum_info import (
    DensityMatrix,
    Pauli,
    SparsePauliOp,
    Statevector,
    partial_trace,
)
from scipy.linalg import ishermitian, logm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, expm, expm_multiply, minres, svds


class HamiltonianLearning:
    def __init__(
        self,
        state: Statevector | QuantumCircuit | DensityMatrix,
        k_learning: int,
        k_constraints: int,
        parameters: None | list[np.ndarray] = None,
        periodic: bool = False,
    ) -> None:
        if isinstance(state, Statevector):
            self.num_qubits = state.num_qubits // 2
            self.state = state
        elif isinstance(state, QuantumCircuit):
            self.num_qubits = state.num_qubits // 2
            self.state = state

        elif isinstance(state, DensityMatrix):
            self.num_qubits = state.num_qubits
            self.state = state

        else:
            print("Not valid state", type(state))

        self.learning_basis = KLocalPauliBasis(
            k=k_learning, num_qubits=self.num_qubits, periodic=periodic
        )
        self.constraint_basis = KLocalPauliBasis(
            k=k_constraints, num_qubits=self.num_qubits, periodic=periodic
        )
        self.sampling_basis = KLocalPauliBasis(
            k=k_learning + k_constraints - 1,
            num_qubits=self.num_qubits,
            periodic=periodic,
        )
        self.sampled_paulis = None
        self.constraint_matrix = None
        self.reconstructed_hamiltonian = None
        self.singular_decomposition = None
        self.parameters = parameters

    def _expectation_value(self, pauli: str) -> float:
        if isinstance(self.state, Statevector):
            pauli_op = Pauli(pauli + "I" * len(pauli))
            return self.state.expectation_value(pauli_op)
        if isinstance(self.state, DensityMatrix):
            return self.state.expectation_value(Pauli(pauli))

    def sample_paulis(self, shots: int = 10000) -> np.array:
        """Creates a dictionary of sampled paulis and their probabilities from a given
        state and pauli basis to sample from."""
        if isinstance(self.state, QuantumCircuit):
            estimator = Estimator()
            observables = [
                Pauli(pauli + "I" * len(pauli))
                for pauli in self.sampling_basis.paulis_list
            ]
            result = estimator.run(
                [self.state.bind_parameters(self.parameters[-1])] * len(observables),
                observables=observables,
                shots=shots,
            ).result()
            self.sampled_paulis = result.values
        else:
            self.sampled_paulis = np.array(
                [
                    self._expectation_value(pauli)
                    for pauli in self.sampling_basis.paulis_list
                ]
            )
        # self.sampled_paulis[np.abs(self.sampled_paulis) < 1e-10] = 0

    def create_constraint_matrix(self) -> np.ndarray:
        """Creates a constraint matrix from the sampled paulis.

        Args:
            sampled_paulis: A dictionary of sampled paulis and their probabilities.
            Aq_basis: A list of k+1 paulis for the q coordinate.
            Sm_basis: A list of k paulis for the m coordinate.
        """
        data = []
        row = []
        col = []
        for i, Aq_label in enumerate(self.constraint_basis.paulis_list):
            Aq_Pauli = Pauli(Aq_label)
            for j, Sm_label in enumerate(self.learning_basis.paulis_list):
                Sm_Pauli = Pauli(Sm_label)
                if Aq_Pauli.anticommutes(Sm_Pauli):
                    operator = 1j * Aq_Pauli @ Sm_Pauli
                    phase = 1j**operator.phase
                    pauli_label = (operator * phase).to_label()
                    value = (
                        phase
                        * self.sampled_paulis[
                            self.sampling_basis.pauli_to_num(pauli_label)
                        ]
                    )
                    if np.abs(value) != 0:
                        row.append(i)
                        col.append(j)
                        data.append(value)
                elif not Aq_Pauli.commutes(Sm_Pauli):
                    raise ValueError("Paulis do not commute or anticommute.")

        self.constraint_matrix = csr_matrix(
            (data, (row, col)),
            shape=(self.constraint_basis.size, self.learning_basis.size),
        )

    ############################

    def reconstruct_hamiltonian(self):
        """Returns a list of the singular values and a list of the singular vectors of the
        constraint matrix."""
        KTK = self.constraint_matrix.T.conj().dot(self.constraint_matrix)
        sing_vals, sing_vecs = np.linalg.eigh(KTK.todense())
        sing_vecs[np.abs(sing_vecs) < 1e-10] = 0
        result = [None] * len(sing_vals)
        for i, _ in enumerate(result):
            result[i] = (sing_vals[i], np.asarray(sing_vecs[:, i]).reshape(-1))
        self.singular_decomposition = result

    def time_evol_faultyH(self):
        norms = []
        H_vecs = []
        for k in range(len(self.parameters)):
            H_vec, norm = self.classical_learn_hamiltonian(k)
            H_vecs.append(H_vec)
            norms.append(norm)
        return H_vecs, norms

    def classical_learn_hamiltonian(self, i: int = -1):
        """Finds the exact hamiltonian leading to the given state."""
        if isinstance(self.state, QuantumCircuit):
            mixed_state = partial_trace(
                Statevector(self.state.bind_parameters(self.parameters[i])),
                range(self.num_qubits),
            )
        if isinstance(self.state, DensityMatrix):
            mixed_state = self.state.data

        hamiltonian_cl_rec = -logm(mixed_state.data)
        hamiltonian_cl_rec = hamiltonian_cl_rec  # - np.eye(hamiltonian_cl_rec.shape[0])*np.trace(hamiltonian_cl_rec)/hamiltonian_cl_rec.shape[0]
        recov_vec = [
            np.trace(hamiltonian_cl_rec @ Pauli(p).to_matrix())
            for p in self.learning_basis.paulis_list
        ]
        recov_vec = np.array([v / hamiltonian_cl_rec.shape[0] for v in recov_vec])
        norm_vec = np.linalg.norm(recov_vec)
        unit_vec = recov_vec / norm_vec if norm_vec != 0 else recov_vec
        return unit_vec, norm_vec

    # def project_hamiltonian(self, l: int, hamiltonian: SparsePauliOp):
    #     """Projects the original hamiltonian onthe the k first singular vectors of the constraint matrix."""
    #     original_vector = self.learning_basis.pauli_to_vector(hamiltonian)
    #     projection = np.zeros_like(self.singular_decomposition[0][1])
    #     projection_coeffs = np.zeros(l)
    #     for i in range(l):
    #         projection_coeffs[i] = np.real(
    #             np.dot(self.singular_decomposition[i][1], original_vector)
    #         )
    #         projection += projection_coeffs[i] * self.singular_decomposition[i][1]
    #     return projection, projection_coeffs
