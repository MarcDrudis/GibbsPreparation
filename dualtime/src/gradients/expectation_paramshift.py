import numpy as np
from qiskit.circuit import QuantumCircuit

from .gradient import Gradient
from ..expectation import Expectation


class ExpectationParameterShift(Gradient):
    def check_setup(self):
        super().check_setup()

        if self.partial is True:
            raise NotImplementedError("Parameter shift does not support complex gradients.")

    def compute(self, values: np.ndarray) -> np.ndarray:
        self.check_setup()

        plus = (values + np.pi / 2 * np.identity(values.size)).tolist()
        minus = (values - np.pi / 2 * np.identity(values.size)).tolist()

        evaluated = self.expectation.evaluate(plus + minus)
        return (evaluated[: values.size] - evaluated[values.size :]) / 2
