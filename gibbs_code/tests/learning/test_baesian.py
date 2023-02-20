import unittest

import numpy as np
from ddt import data, ddt, unpack
from gibbs.learning.bayesian_learning import BayesianLearning
from gibbs.learning.constraint_matrix import ConstraintMatrixFactory
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.utils import simple_purify_hamiltonian
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector, partial_trace


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
        control_fields = [np.zeros(self.basis.size)] + [cfield] * 3

        self.bl = BayesianLearning(
            states=states,
            control_fields=control_fields,
            constraint_matrix_factory=ConstraintMatrixFactory(
                self.num_qubits, self.k, self.k
            ),
            prior_mean=coriginal,
            prior_covariance=(1, 0.1),
            sampling_std=0.0001,
            shots=1e1,
        )

    def test_construction(self):
        self.SetUp()
        c_test = np.zeros(self.basis.size * len(self.bl.control_fields))
        c_test[0:5] = 1
        cond_cov = self.bl.cond_covariance(c_test, [0, 1])
        # print(cond_cov)

    # def test_update(self):
    #     self.SetUp()
    #     print(self.bl.)

    def test_partial_cov(self):
        self.SetUp()
        indexes = [0, 2]

        assert self.bl.partial_cov(indexes).shape == (
            self.basis.size * len(indexes),
            self.basis.size * len(indexes),
        ), f"The shapes are not as expected. We get {self.bl.partial_cov(indexes).shape} and should get square with {self.basis.size * len(indexes)}"
