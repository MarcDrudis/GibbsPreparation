from __future__ import annotations

import numpy as np
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector
from scipy.sparse import csr_matrix
from itertools import product


class ConstraintMatrixFactory:
    def __init__(
        self,
        num_qubits: int,
        k_learning: int,
        k_constraints: int,
        periodic: bool = False,
        estimator: Estimator = Estimator(),
    ) -> None:
        """
        Args:
            num_qubits: Number of qubits in our Hamiltonian.
            k_learning: k-locality on the basis that we want to reconstruct our Hamiltonian from.
            k_constraints: k-locality for the basis of the constraints we want to use (typically k+1).
            periodic: Whether our local terms are periodic.
        """
        self.num_qubits = num_qubits
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
        self.counter_shots = 0
        self.estimator = estimator

    def _expectation_value(
        self, state: Statevector | DensityMatrix, pauli: str, shots: int | None
    ) -> float:
        if shots is not None:
            self.counter_shots += shots
        if isinstance(state, Statevector):
            obs = Pauli(pauli + "I" * len(pauli))
        if isinstance(state, DensityMatrix):
            obs = Pauli(pauli)
        expectation_value = np.real_if_close(state.expectation_value(obs))
        if shots is None:
            return expectation_value, 0
        else:
            sq_obs = obs @ obs
            sq_exp_val = np.real_if_close(state.expectation_value(sq_obs))
            variance = sq_exp_val - expectation_value**2
            variance = max(variance, 0)
            standard_deviation = np.sqrt(variance / shots)
            return (
                np.random.normal(expectation_value, standard_deviation),
                variance,
            )

    def sample_paulis(
        self, state: QuantumCircuit | Statevector, shots: int
    ) -> tuple[list[float], list[float]]:
        """Creates a dictionary of sampled paulis and their probabilities from a given
        state and pauli basis to sample from."""
        if isinstance(state, QuantumCircuit):
            observables = [
                Pauli(pauli + "I" * len(pauli))
                for pauli in self.sampling_basis.paulis_list
            ]
            result = self.estimator.run(
                [state] * len(observables),
                observables=observables,
                shots=shots,
            ).result()
            values = result.values
            variances = [m["variance"] for m in result.metadata]
            return values, variances

        elif isinstance(state, (Statevector, DensityMatrix)):
            result = [
                self._expectation_value(state, pauli, shots)
                for pauli in self.sampling_basis.paulis_list
            ]
            return zip(*result)

        else:
            raise AssertionError(f"Wrong state type to sample from: {type(state)}")

    def create_cmat_from_sample(
        self, sampled_paulis: list, variances: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """Creates a constraint matrix from the sampled paulis."""
        con_len = self.constraint_basis.size
        learn_len = self.learning_basis.size
        sampl_len = self.sampling_basis.size
        K_mat = np.zeros((con_len, learn_len))
        E_mat = np.zeros(
            (
                con_len,
                learn_len,
                sampl_len,
            )
        )
        for k, i in product(range(con_len), range(learn_len)):
            constraint_pauli = Pauli(self.constraint_basis.num_to_pauli(k))
            learning_pauli = Pauli(self.constraint_basis.num_to_pauli(i))
            if constraint_pauli.anticommutes(learning_pauli):
                operator = 1j * constraint_pauli @ learning_pauli
                phase = 1j**operator.phase
                pauli_label = (operator * phase).to_label()
                index = self.sampling_basis.pauli_to_num(pauli_label)
                val = np.real_if_close(phase * sampled_paulis[index])
                std = variances[index]
                K_mat[k, i] = val
                E_mat[k, i, index] = std
        return K_mat, E_mat

    def create_cmat(
        self, state: QuantumCircuit | Statevector, shots: int = 10000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Samples and creates the constraint matrix."""
        sampled_paulis, variances = self.sample_paulis(state, shots)
        return self.create_cmat_from_sample(sampled_paulis, variances)


# class DumbConstraintMatrixFactory(ConstraintMatrixFactory):
#     def create_cmat(
#         self, state: QuantumCircuit | Statevector, shots: int = 10000
#     ) -> np.ndarray:
#         """Creates a constraint matrix from the sampled paulis.

#         Args:
#             sampled_paulis: A dictionary of sampled paulis and their probabilities.
#             Aq_basis: A list of k+1 paulis for the q coordinate.
#             Sm_basis: A list of k paulis for the m coordinate.
#         """
#         data = []
#         row = []
#         col = []
#         for i, Aq_label in enumerate(self.constraint_basis.paulis_list):
#             Aq_Pauli = Pauli(Aq_label)
#             for j, Sm_label in enumerate(self.learning_basis.paulis_list):
#                 Sm_Pauli = Pauli(Sm_label)
#                 if i < j:
#                     break
#                 if Aq_Pauli.anticommutes(Sm_Pauli):
#                     value = self._get_value(Aq_Pauli, Sm_Pauli, state, shots)
#                     if np.abs(value) != 0:
#                         row.append(i)
#                         col.append(j)
#                         data.append(value)
#                 elif not Aq_Pauli.commutes(Sm_Pauli):
#                     raise ValueError("Paulis do not commute or anticommute.")

#         mat = csr_matrix(
#             (data, (row, col)),
#             shape=(self.constraint_basis.size, self.learning_basis.size),
#         )
#         mat[: mat.shape[1], :] -= mat[: mat.shape[1], :].T
#         return mat

#     def _get_value(self, Aq_Pauli: Pauli, Sm_Pauli: Pauli, state, shots):
#         """Gets the value of a given commutator from a list of presampled paulis.
#         This avoids havning to sample the same pauli many times."""
#         operator = 1j * Aq_Pauli @ Sm_Pauli
#         phase = 1j**operator.phase
#         pauli_label = (operator * phase).to_label()
#         sampled_pauli = self._expectation_value(state, pauli_label, shots)
#         return phase * sampled_pauli
