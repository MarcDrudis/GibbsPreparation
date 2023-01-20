from typing import Optional

from abc import ABC, abstractmethod
import numpy as np

from ..expectation import Expectation


class Gradient(ABC):
    """Gradients of expectation values."""

    def __init__(self, expectation: Optional[Expectation] = None, partial: bool = False) -> None:
        r"""
        Args:
            expectation: The expectation object to compute the gradient for. Can be supplied
                after the initialization.
            partial: If True, return :math:`\rangle \partial_j \phi(\theta) | H | \phi(\theta)`,
                else twice the real part of the above (= the gradient of the expectation).
        """
        self.expectation = expectation
        self.partial = partial

    def check_setup(self) -> None:
        """Check if setup is valid, i.e. if the expectation is set."""
        if self.expectation is None:
            raise ValueError("Expectation is not set.")

    @abstractmethod
    def compute(self, values: np.ndarray) -> np.ndarray:
        """Compute the expectation value gradient.

        Args:
            values: The values at which the gradient is computed.

        Returns:
            The gradient of the expectation value at ``values``.
        """
        raise NotImplementedError
