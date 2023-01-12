from __future__ import annotations
import numpy as np
from itertools import product
import functools
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp, partial_trace, DensityMatrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import expm_multiply, minres, eigsh, expm
from scipy.linalg import ishermitian
from qiskit import QuantumCircuit
from gibbs.learning.klocal_hamiltonian import KLocalPauliBasis


class HamiltonianLearning:
    
    def __init__(self,state: Statevector|QuantumCircuit|DensityMatrix , k_learning:int, k_constraints:int) -> None:
        
        if isinstance(state,Statevector):
            self.num_qubits = state.num_qubits//2
        if isinstance(state,QuantumCircuit):
            self.num_qubits = state.num_qubits//2
        if isinstance(state,DensityMatrix):
            self.num_qubits = state.num_qubits
        
        self.state = state
        self.learning_basis = KLocalPauliBasis(k=k_learning, num_qubits = self.num_qubits)
        self.constraint_basis = KLocalPauliBasis(k=k_constraints, num_qubits = self.num_qubits)
        self.sampling_basis = KLocalPauliBasis(k=k_learning+k_constraints-1, num_qubits = self.num_qubits)
        self.sampled_paulis = None
        self.constraint_matrix = None
        self.reconstructed_hamiltonian = None
    
    def _expectation_value(self, pauli: str) -> float:
        if isinstance(self.state,Statevector):
            pauli_op = Pauli(pauli + "I" * len(pauli))
            return self.state.expectation_value(pauli_op)
        if isinstance(self.state,DensityMatrix):
            return self.state.expectation_value(Pauli(pauli))
                
    def sample_paulis(self) -> np.array:
        """Creates a dictionary of sampled paulis and their probabilities from a given
        state and pauli basis to sample from."""
        self.sampled_paulis = np.array([self._expectation_value(pauli) for pauli in self.sampling_basis._paulis_list])
        self.sampled_paulis[np.abs(self.sampled_paulis) < 1e-10] = 0
    
        

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
        for i, Aq_label in enumerate(self.constraint_basis._paulis_list):
            Aq_Pauli = Pauli(Aq_label)
            for j, Sm_label in enumerate(self.learning_basis._paulis_list):
                Sm_Pauli = Pauli(Sm_label)
                if Aq_Pauli.anticommutes(Sm_Pauli):
                    operator = 1j * Aq_Pauli @ Sm_Pauli
                    phase = 1j**operator.phase
                    pauli_label = (operator * phase).to_label()
                    value = phase * self.sampled_paulis[self.sampling_basis.pauli_to_num(pauli_label)]
                    if np.abs(value) != 0:
                        row.append(i)
                        col.append(j)
                        data.append(value)
                elif not Aq_Pauli.commutes(Sm_Pauli):
                    raise ValueError("Paulis do not commute or anticommute.")

        self.constraint_matrix = csr_matrix((data, (row, col)), shape=(self.constraint_basis.size, self.learning_basis.size))

    def reconstruct_hamiltonian(self):
        """Returns a list of the singular values and a list of the singular vectors of the
        constraint matrix."""
        KTK = self.constraint_matrix.T.conj().dot(self.constraint_matrix)
        sing_vals, sing_vecs = np.linalg.eigh(KTK.todense())
        sing_vecs[np.abs(sing_vecs) < 1e-10] = 0
        result = [None]*len(sing_vals)
        for i,_ in enumerate(result):
            result[i] = (sing_vals[i],sing_vecs[:,i].T)
            
        return result
