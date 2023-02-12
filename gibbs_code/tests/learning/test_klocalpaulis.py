import unittest

import numpy as np
from ddt import data, ddt, unpack
from gibbs.learning.hamiltonian_learning import HamiltonianLearning
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.utils import (
    create_hamiltonian_lattice,
    identity_purification,
    printarray,
    simple_purify_hamiltonian,
)
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector, partial_trace
from scipy.sparse.linalg import expm, expm_multiply


@ddt
class TestKLocalPauliBasis(unittest.TestCase):
    def test_basis(self):
        pbs = KLocalPauliBasis(2, 4)
        print(pbs.paulis_list)
        for pauli in create_hamiltonian_lattice(4, 1, 1).paulis:
            assert (
                pauli.to_label() in pbs.paulis_list
            ), f"Not in the basis: {pauli.to_label()}"

    @data(
        ["XX", 5, ["XXIII", "IXXII", "IIXXI", "IIIXX", "XIIIX"]],
    )
    @unpack
    def test_extend_pauli(self, pauli_str, num_qubits, expected):
        basis = KLocalPauliBasis(len(pauli_str), num_qubits, True)
        result = basis._extend_pauli(pauli_str)
        np.testing.assert_equal(result, expected)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
