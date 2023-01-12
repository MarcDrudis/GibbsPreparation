import unittest
from ddt import data, ddt, unpack
from qiskit.circuit import QuantumCircuit,Parameter
from gibbs.learning.hamiltonian_learning import HamiltonianLearning
from qiskit.quantum_info import Statevector,SparsePauliOp,partial_trace, DensityMatrix
from scipy.sparse.linalg import expm_multiply, expm
import numpy as np
from gibbs.utils import identity_purification, printarray, create_hamiltonian_lattice, simple_purify_hamiltonian

@ddt
class TestHamiltonianLearning(unittest.TestCase):
    
    def test_sampling(self):
        state = Statevector.from_label("0"*12)
        hl = HamiltonianLearning(state,2,3)
        hl.sample_paulis()
        # print(hl.sampled_paulis)
        
        
    def test_constraint_matrix(self):
        state = Statevector.from_label("0"*8)
        hl = HamiltonianLearning(state,2,3)
        hl.sample_paulis()
        hl.create_constraint_matrix()
        assert hl.constraint_matrix.shape == (111, 39)
    
    @data(
        [create_hamiltonian_lattice(4,1,1)],
        [create_hamiltonian_lattice(4,-1,1)],
        [create_hamiltonian_lattice(4,10,0.01)]
    )
    @unpack
    def test_reconstruct_hamiltonian(self,original_hamiltonian):
        N = original_hamiltonian.num_qubits
        state = simple_purify_hamiltonian(original_hamiltonian, noise=0)
        hl = HamiltonianLearning(state,2,3)
        
        hl.sample_paulis()
        hl.create_constraint_matrix()
        result = hl.reconstruct_hamiltonian()
        
        original_vector = hl.learning_basis.pauli_to_vector(original_hamiltonian)
        original_vector = original_vector/np.linalg.norm(original_vector)
        normalized_result = result[0][1]/np.linalg.norm(result[0][1])
        
        error = min(np.linalg.norm(result[0][1] + original_vector.T), np.linalg.norm(result[0][1] - original_vector.T))
        assert error < 1e-7, "Reconstruction failed with error: {}".format(error)
        
        
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)