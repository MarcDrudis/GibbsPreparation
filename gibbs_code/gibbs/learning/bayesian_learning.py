from __future__ import annotations
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit import QuantumCircuit
from gibbs.learning.constraint_matrix import ConstraintMatrixFactory
from scipy.linalg import block_diag
from scipy.optimize import minimize
import numpy as np


class BayesianLearning:
    def __init__(
        self,
        states: list[Statevector | QuantumCircuit],
        control_fields: list[np.ndarrays],
        constraint_matrix_factory: ConstraintMatrixFactory,
        prior_mean: np.ndarray,
        prior_c_cov: np.ndarray,
        prior_cfield_cov: np.ndarray,
        sampling_std: float,
        prior_preparation_noise: float,
    ) -> None:
        """
        Args:
            states: Series of states to sample from.
            control_fields: Control fields associated to each one of the prepared states.
        """
        self.states = states
        self.control_fields = control_fields
        self.constraint_matrix_factory = constraint_matrix_factory
        self.current_mean = prior_mean
        self.sampling_std = sampling_std
        self.prior_preparation_noise = prior_preparation_noise

        self.current_cov = prior_c_cov
        self.cfield_cov = prior_cfield_cov
        
    @property
    def current_inv_cov(self):
        return np.linalg.inv(self.current_cov)
    
    @property
    def cfield_inv_cov(self):
        return np.linalg.inv(self.cfield_cov)

    def cond_covariance(self, c, cfield_index) -> np.ndarray:
        cov = self.sampling_std * (
            self.current_cov
            + self.cfield_cov
            + np.outer(c, c)
            + np.outer(
                self.control_fields[cfield_index], self.control_fields[cfield_index]
            )
        )
        np.fill_diagonal(
            cov,
            np.trace(self.current_cov)
            + np.trace(self.cfield_cov)
            + np.dot(c, c)
            + np.dot(
                self.control_fields[cfield_index], self.control_fields[cfield_index]
            ),
        )
        
        return cov
    
    def _cost_function(self,c,cfield_index,A,Lx):
        x = np.append(c,self.control_fields[cfield_index])
        Lex = np.linalg.cholesky(self.cond_covariance(c,cfield_index))
        print(Lex.shape)
        print(A.shape)
        print(x.shape)
        first_term = np.linalg.norm(Lex@A@(c+self.control_fields[cfield_index]))**2
        second_term = np.linalg.norm(Lx@(c-self.current_mean))**2
        return first_term + second_term
    
        
    def update(self,cfield_index):
        Lx = np.linalg.cholesky(block_diag(self.current_inv_cov,self.cfield_inv_cov))
        constraint_matrix = self.constraint_matrix_factory.create_constraint_matrix(self.states[cfield_index])
        cost_function = lambda c : self._cost_function(c,cfield_index,constraint_matrix,Lx)
        posterior_mean = minimize(cost_function,self.current_mean)
        return posterior_mean