import numpy as np
from qiskit.circuit import QuantumCircuit

from surfer.gradient import ReverseGradient

from .gradient import Gradient
from ..expectation import Expectation


class ExpectationReverse(Gradient):
    """Expectation value gradient with the classically fast reverse mode."""

    def compute(self, values: np.ndarray) -> np.ndarray:
        self.check_setup()

        circuit = self.expectation.circuit
        op = self.expectation.hamiltonian
        return ReverseGradient(partial_gradient=self.partial).compute(op, circuit, values)
