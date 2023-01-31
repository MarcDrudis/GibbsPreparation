import unittest
from ddt import data, ddt, unpack
from gibbs.learning.bayesian_learning import BayesianLearning
from gibbs.learning.constraint_matrix import ConstraintMatrixFactory
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from qiskit.quantum_info import Statevector, SparsePauliOp, partial_trace, DensityMatrix
import numpy as np
from gibbs.utils import simple_purify_hamiltonian


@ddt
class TestBayesianLearning(unittest.TestCase):
    def SetUp(self):
        self.k = 2
        self.num_qubits = 8
        self.basis = KLocalPauliBasis(self.k, self.num_qubits)
        coriginal = np.zeros(self.basis.size)
        coriginal[:5] = 0.5
        cfield = np.zeros(self.basis.size)
        cfield[4:7] = -0.25
        states = [
            simple_purify_hamiltonian(self.basis.vector_to_pauli_op(vec))
            for vec in [coriginal, coriginal + cfield]
        ]
        control_fields = [np.zeros(self.basis.size), cfield]
        prior_c_cov = np.eye(self.basis.size) * 0.1
        prior_cfield_cov = np.eye(self.basis.size) * 0.01
        self.bl = BayesianLearning(
            states=states,
            control_fields=control_fields,
            constraint_matrix_factory=ConstraintMatrixFactory(
                self.num_qubits, self.k, self.k
            ),
            prior_mean=coriginal,
            prior_c_cov=prior_c_cov,
            prior_cfield_cov=prior_cfield_cov,
            sampling_std=1,
            prior_preparation_noise=1e-3,
        )

    def test_construction(self):
        self.SetUp()
        c_test = np.zeros(self.basis.size)
        c_test[0:5] = 1
        cond_cov = self.bl.cond_covariance(c_test, 0)
        # print(cond_cov)


    def test_update(self):
        self.SetUp()
        print(self.bl.update(0))