from __future__ import annotations
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit import QuantumCircuit
from gibbs.learning.constraint_matrix import ConstraintMatrixFactory
from scipy.linalg import block_diag
from scipy.sparse import bmat
from scipy.optimize import minimize
import numpy as np
import time


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
    ) -> None:
        """
        Args:
            states: Series of states to sample from.
            control_fields: Control fields associated to each one of the prepared states.
        """
        self.states = states
        self.control_fields = control_fields
        self.constraint_matrix_factory = constraint_matrix_factory
        self.sampling_std = sampling_std

        self.current_mean = prior_mean
        self.current_cov = prior_c_cov
        self.cfield_cov = prior_cfield_cov

        self.constraint_matrices = None

    @property
    def current_inv_cov(self):
        return np.linalg.inv(self.current_cov)

    @property
    def cfield_inv_cov(self):
        return np.linalg.inv(self.cfield_cov)

    def constraint_matrix(self, index: int) -> np.ndarray:
        if self.constraint_matrices == None:
            self.constraint_matrices = [
                self.constraint_matrix_factory.create_constraint_matrix(state)
                for state in self.states
            ]
        return self.constraint_matrices[index]

    def cond_covariance(self, c,v) -> np.ndarray:
        """
        This returns a block of the conditional covariance. The shape should be the same as c.
        """
        cfield_cov = self.cfield_cov
        cov = (
            self.current_cov
            + cfield_cov
            + np.outer(c, c)
            + np.outer(v, v)
        )
        
        np.fill_diagonal(
            cov,
            np.trace(self.current_cov)
            + np.trace(cfield_cov)
            + np.dot(c, c)
            + np.dot(v, v)
        )

        return self.sampling_std * cov

    def _cost_function(
        self, x, cfield_index, A, Lx
    ):  # Here I really need to check whether I am using x or c in the correct way
        # The main concern is that should I be changing
        c = x[: x.size // 2]
        v = x[x.size // 2 :]
        
        Gammaex = block_diag(
            self.cond_covariance(c,np.zeros_like(c)), self.cond_covariance(c,v)
        )
        
        Lex = np.linalg.cholesky(np.linalg.inv(Gammaex))
        
        xbar = np.append(self.current_mean,self.control_fields[cfield_index])
        loss = np.linalg.norm(Lex @A@ x) ** 2
        regularization = (
            np.linalg.norm(Lx @ (x - xbar)) ** 2
        )  # Specially here
        # print(loss,regularization)
        return loss + regularization

    def update_mean(self, cfield_index):
        tiempoa = time.time()
        Lx = np.linalg.cholesky(block_diag(self.current_inv_cov, self.cfield_inv_cov))
        constraint_matrix = bmat(
            [
                [   self.constraint_matrix(0),
                    None
                ],
                [
                    self.constraint_matrix(cfield_index),
                    self.constraint_matrix(cfield_index)
                ],
            ]
        )
        
        cost_function = lambda x: self._cost_function(
            x, cfield_index, constraint_matrix, Lx
        )
        x0 = np.append(self.current_mean, self.control_fields[cfield_index])
        
        assert x0.size == constraint_matrix.shape[1], f"Size of x0{x0.size} and constraint matrix {constraint_matrix.shape} don't match"
        
        tiempob=time.time()
        options = {"xatol":1e-5,"disp":True,"maxiter":1e5}
        posterior_mean = minimize(cost_function, x0, method="Nelder-Mead",options=options).x
        # from cma import fmin
        # posterior_mean = fmin(cost_function,x0,1,)[0]
        tiempoc= time.time()
        print("The time it takes for minimize is:",tiempoc-tiempob,"for the rest:",tiempob-tiempoa)
        return posterior_mean
    
    def update_cov(self,x,cfield_index):
        c = x[: x.size // 2]
        v = x[x.size // 2 :]
        constraint_matrix = bmat(
            [
                [   self.constraint_matrix(0),
                    None
                ],
                [
                    self.constraint_matrix(cfield_index),
                    self.constraint_matrix(cfield_index)
                ],
            ]
        )
        Gammaex = block_diag(
            self.cond_covariance(c, np.zeros_like(c)), self.cond_covariance(c, v)
        )
        invgammaex = np.linalg.inv(Gammaex)
        return np.linalg.inv(block_diag(self.current_inv_cov, self.cfield_inv_cov)+  constraint_matrix.T@invgammaex@constraint_matrix)  