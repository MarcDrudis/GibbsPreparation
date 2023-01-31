from __future__ import annotations
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from qiskit.quantum_info import Statevector, DensityMatrix, Pauli
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit
import numpy as np
from scipy.sparse import csr_matrix


class ConstraintMatrixFactory:
    def __init__(
        self,
        num_qubits: int,
        k_learning: int,
        k_constraints: int,
        periodic: bool = False,
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

    def _expectation_value(
        self, state: Statevector | DensityMatrix, pauli: str
    ) -> float:
        if isinstance(state, Statevector):
            pauli_op = Pauli(pauli + "I" * len(pauli))
            return state.expectation_value(pauli_op)
        if isinstance(state, DensityMatrix):
            return state.expectation_value(Pauli(pauli))

    def sample_paulis(
        self, state: QuantumCircuit | Statevector, shots: int
    ) -> np.array:
        """Creates a dictionary of sampled paulis and their probabilities from a given
        state and pauli basis to sample from."""
        if isinstance(state, QuantumCircuit):
            estimator = Estimator()
            observables = [
                Pauli(pauli + "I" * len(pauli))
                for pauli in self.sampling_basis._paulis_list
            ]
            result = estimator.run(
                [state.bind_parameters(self.parameters[-1])] * len(observables),
                observables=observables,
                shots=shots,
            ).result()
            return result.values
        elif isinstance(state, (Statevector,DensityMatrix)):
            return np.array(
                [
                    self._expectation_value(state,pauli)
                    for pauli in self.sampling_basis._paulis_list
                ]
            )
        else:
            raise AssertionError(f"Wrong state type to sample from: {type(state)}")

    def create_constraint_matrix(
        self, state: QuantumCircuit | Statevector, shots: int = 10000
    ) -> np.ndarray:
        """Creates a constraint matrix from the sampled paulis.

        Args:
            sampled_paulis: A dictionary of sampled paulis and their probabilities.
            Aq_basis: A list of k+1 paulis for the q coordinate.
            Sm_basis: A list of k paulis for the m coordinate.
        """
        sampled_paulis = self.sample_paulis(state, shots)
        data = []
        row = []
        col = []
        for i, Aq_label in enumerate(self.constraint_basis._paulis_list):
            Aq_Pauli = Pauli(Aq_label)
            for j, Sm_label in enumerate(self.learning_basis._paulis_list):
                Sm_Pauli = Pauli(Sm_label)
                if Aq_Pauli.anticommutes(Sm_Pauli):
                    operator = 1j * Aq_Pauli @ Sm_Pauli
                    phase = 1j**operator.phase
                    pauli_label = (operator * phase).to_label()
                    value = (
                        phase
                        * sampled_paulis[self.sampling_basis.pauli_to_num(pauli_label)]
                    )
                    if np.abs(value) != 0:
                        row.append(i)
                        col.append(j)
                        data.append(value)
                elif not Aq_Pauli.commutes(Sm_Pauli):
                    raise ValueError("Paulis do not commute or anticommute.")

        return csr_matrix(
            (data, (row, col)),
            shape=(self.constraint_basis.size, self.learning_basis.size),
        )
