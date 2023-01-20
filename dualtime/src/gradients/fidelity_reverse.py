from __future__ import annotations
import numpy as np
from qiskit.opflow import MatrixOp

from surfer.gradient import ReverseGradient

from .fidelity_gradient import FidelityGradient
from ..fidelity import Fidelity


class FidelityReverse(FidelityGradient):
    def __init__(self, fidelity: Fidelity | None = None, use_local_projector: bool = False) -> None:
        super().__init__(fidelity)

        self.use_local_projector = use_local_projector

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute the fidelity gradient, where ``values`` is the first argument."""
        self.check_setup()

        if isinstance(left, list) and isinstance(right, list):
            return [self.compute(left_i, right_i) for left_i, right_i in zip(left, right)]

        left_circuit = self.fidelity.left_circuit
        right_circuit = self.fidelity.right_circuit

        circuit = right_circuit.compose(left_circuit.bind_parameters(left).inverse())

        if self.use_local_projector:
            raise NotImplementedError()
        else:
            proj = MatrixOp(np.diag([1] + [0] * (2**right_circuit.num_qubits - 1)))

        grad = ReverseGradient().compute(proj, circuit, right)
        return grad
