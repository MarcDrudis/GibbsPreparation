import unittest

import numpy as np

# pylint: disable=import-error, no-name-in-module
from qiskit.providers.aer import AerSimulator, StatevectorSimulator

from qiskit.opflow import AerPauliExpectation
from qiskit.circuit.library import RealAmplitudes
from src.fidelity import Fidelity
from src.gradients import FidelityParameterShift, FidelityReverse


class TestFidelityGradient(unittest.TestCase):
    def test_gradients(self):
        """Test gradients are the same."""
        circuit = RealAmplitudes(2, reps=1)
        backend = AerSimulator(method="statevector")
        expectation = AerPauliExpectation()
        fidelity = Fidelity(backend, expectation, circuit, circuit)
        theta = np.arange(1, circuit.num_parameters + 1) / circuit.num_parameters
        next_theta = theta + np.random.random(circuit.num_parameters)

        paramshift = FidelityParameterShift(fidelity).compute(theta, next_theta)
        exact = FidelityReverse(fidelity).compute(theta, next_theta)

        np.testing.assert_allclose(paramshift, exact)
