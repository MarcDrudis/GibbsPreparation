from typing import Optional
from abc import ABC, abstractmethod
import numpy as np

from ..fidelity import Fidelity


class FidelityGradient(ABC):
    def __init__(self, fidelity: Optional[Fidelity] = None) -> None:
        self.fidelity = fidelity

    def check_setup(self) -> None:
        if self.fidelity is None:
            raise ValueError("Fidelity is not set.")

    @abstractmethod
    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raise NotImplementedError
