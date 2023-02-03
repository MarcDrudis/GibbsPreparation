from __future__ import annotations
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit import QuantumCircuit
from gibbs.learning.constraint_matrix import ConstraintMatrixFactory
from scipy.linalg import block_diag
from scipy.sparse import bmat
from scipy.optimize import minimize
from gibbs.utils import classical_learn_hamiltonian
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
        shots:int,
    ) -> None:
        """
        Args:
            states: Series of states to sample from.
            control_fields: Control fields associated to each one of the prepared states.
        """
        self.states = states
        self.shots = shots
        self.control_fields = control_fields
        self.constraint_matrix_factory = constraint_matrix_factory
        self.sampling_std = sampling_std

        self.current_mean = prior_mean
        self.total_cov = None
        self.prior_cov = prior_c_cov
        self.prior_cfield_cov = prior_cfield_cov

        self.constraint_matrices = None
        self.display = True

    
    @property
    def current_cov(self):
        """Covariance of c, what we are interested. This is only used to compute the blocks in the conditional covariance"""
        if self.total_cov is None:
            return self.prior_cov
        return self.total_cov[:self.total_cov.shape[0]//2,:self.total_cov.shape[1]//2]
    
    @property
    def cfield_cov(self):
        """Covariance of v, the control field. We update the knowledge we had on the control fields as well.
        This is only used to compute the blocks in the conditional covariance"""
        if self.total_cov is None:
            return self.prior_cfield_cov
        return self.total_cov[self.total_cov.shape[0]//2:,self.total_cov.shape[1]//2:]
    
    @property
    def inverse_cov(self):
        """Inverse of the current covariance matrix"""
        if self.total_cov is None:
            return block_diag(np.linalg.inv(self.prior_cov),np.linalg.inv(self.prior_cfield_cov))
        return np.linalg.inv(self.total_cov)
    @property
    def Lx(self):
        return np.linalg.cholesky(self.inverse_cov)

    def constraint_matrix(self, index: int) -> np.ndarray:
        if self.constraint_matrices == None:
            self.constraint_matrices = [
                self.constraint_matrix_factory.create_constraint_matrix(state,shots=self.shots)
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
        # if self.display:
        #     self.display = False
        #     print("Lx")
        #     print(Lx.diagonal())
        #     print("Lex")
        #     print(Lex.diagonal())
        return loss + regularization
    
    @staticmethod
    def block_control_matrix(constraint_matrices:list):
        return bmat([[constraint_matrices[0]]+[None]*(len(constraint_matrices)-1)]+[[c]+[None]*i + [c] +[None]*(len(constraint_matrices)-i-2) for i,c in enumerate(constraint_matrices[1:])])

    def update_mean(self, cfield_index):
        tiempoa = time.time()

        A = self.block_control_matrix([self.constraint_matrix(0),self.constraint_matrix(cfield_index)])
        
        def cost_function(x):
            cost = self._cost_function(x, cfield_index, A, self.Lx)
            # print(cost)
            return cost
        
        x0 = np.append(classical_learn_hamiltonian(self.states[0],2), self.control_fields[cfield_index])
        
        assert x0.size == A.shape[1], f"Size of x0: {x0.size} and constraint matrix: {A.shape} don't match"
        
        tiempob=time.time()
        posterior_mean = minimize(cost_function,x0,options={"maxiter":1e5,"xrtol":1e-5}).x
        # from cma import fmin
        # posterior_mean = fmin(cost_function,x0,1,)[0]
        tiempoc= time.time()
        print("The time it takes for minimize is:",tiempoc-tiempob,"for the rest:",tiempob-tiempoa)
        print(f"The cost function ends up with a value of:{cost_function(posterior_mean)}, it started with a value of {cost_function(x0)}")
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
        return np.linalg.inv(self.inverse_cov+  constraint_matrix.T@invgammaex@constraint_matrix)  