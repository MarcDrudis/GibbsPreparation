import numpy as np

from .fidelity_gradient import FidelityGradient


class FidelityParameterShift(FidelityGradient):
    """The fidelity gradient using the parameter shift rule."""

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        self.check_setup()

        if isinstance(left, list) and isinstance(right, list):
            return [self.compute(left_i, right_i) for left_i, right_i in zip(left, right)]

        dim = right.size
        plus = (right + np.pi / 2 * np.identity(dim)).tolist()
        minus = (right - np.pi / 2 * np.identity(dim)).tolist()

        evaluated = self.fidelity.evaluate([left] * 2 * dim, plus + minus)
        return (evaluated[:dim] - evaluated[dim:]) / 2
