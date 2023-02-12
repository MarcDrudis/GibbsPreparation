from __future__ import annotations
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.circuit import QuantumCircuit
from gibbs.learning.constraint_matrix import ConstraintMatrixFactory
from scipy.linalg import block_diag
from scipy.sparse import bmat
from scipy.optimize import minimize
from gibbs.utils import classical_learn_hamiltonian
from scipy.linalg import cholesky, solve_triangular
from gibbs.utils import printarray
import numpy as np
import time


class BayesianLearning:
    def __init__(
        self,
        states: list[Statevector | QuantumCircuit],
        control_fields: list[np.ndarrays],
        constraint_matrix_factory: ConstraintMatrixFactory,
        prior_mean: np.ndarray,
        prior_covariance: np.ndarray | tuple[float, float],
        sampling_std: float,
        shots: int,
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
        self.size = prior_mean.size
        self.current_mean = prior_mean
        if isinstance(prior_covariance, tuple):
            ones = np.ones(self.size)
            self.total_cov = np.diag(
                np.concatenate(
                    [ones * prior_covariance[0]]
                    + [ones * prior_covariance[0]] * (len(control_fields) - 1)
                )
            )
        else:
            self.total_cov = prior_covariance

        self.constraint_matrices = None

    @property
    def current_cov(self):
        """Covariance of c, what we are interested. This is only used to compute the blocks in the conditional covariance"""
        return self.total_cov[: self.size, : self.size]

    def cfield_cov(self, index: int):
        """Covariance of v, the control field. We update the knowledge we had on the control fields as well.
        This is only used to compute the blocks in the conditional covariance"""
        if index == 0:
            return np.zeros((self.size, self.size))
        return self.total_cov[
            self.size * index : self.size * (index + 1),
            self.size * index : self.size * (index + 1),
        ]

    def partial_cov(self, indexes: list[int]):
        print(indexes)
        pcov = np.split(
            self.total_cov,
            [i * self.size for i in range(len(self.control_fields))],
            axis=0,
        )
        print(len(pcov))
        pcov = np.concatenate([pcov[i] for i in indexes])
        print("why", pcov.shape)

        pcov = np.split(
            self.total_cov,
            [i * self.size for i in range(len(self.control_fields))],
            axis=1,
        )
        pcov = [pcov[i] for i in indexes]
        return pcov

    @property
    def inverse_cov(self):
        """Inverse of the current covariance matrix"""
        return np.linalg.inv(self.total_cov)

    # @property
    # def Lx(self):
    #     return np.linalg.cholesky(self.inverse_cov)

    def constraint_matrix(self, index: int) -> np.ndarray:
        if self.constraint_matrices == None:
            self.constraint_matrices = [
                self.constraint_matrix_factory.create_constraint_matrix(
                    state, shots=self.shots
                )
                for state in self.states
            ]
        return self.constraint_matrices[index]

    def cond_covariance(self, x, indexes: list[int]) -> np.ndarray:
        """
        Returns the covariance of the error conditioned on x.
        """
        cov = block_diag(*[self._single_block_cond_cov(x, i) for i in indexes])
        return cov

    def _single_block_cond_cov(self, x, c_field_index):
        """
        This returns a block of the conditional covariance. The shape should be the same as c.
        """
        cfield_cov = self.cfield_cov(c_field_index)
        c = x[: self.size]
        v = (
            x[self.size * c_field_index : self.size * (c_field_index + 1)]
            if c_field_index != 0
            else np.zeros_like(c)
        )
        cov = self.current_cov + cfield_cov + np.outer(c, c) + np.outer(v, v)
        np.fill_diagonal(
            cov,
            np.trace(self.current_cov)
            + np.trace(cfield_cov)
            + np.dot(c, c)
            + np.dot(v, v),
        )

        return self.sampling_std * cov

    @staticmethod
    def _cond_term(Gammaex, A, x):
        """Given Gammaex, A and x compute the term
        Lex@A@x , where Lex is the cholesky decompostion of Gammaex inverse.
        We use the fact that cholesky of the inverse is the inverse of cholesky.
        Be warry that scipy uses different notation than paper.
        """
        print(Gammaex.shape, A.shape, x.shape)
        L_inv = cholesky(Gammaex, lower=True)
        c = solve_triangular(L_inv, A @ x)
        return np.linalg.norm(c) ** 2

    @staticmethod
    def _distribution_term(Gammax, x, xbar):
        """Returns the other term to minimize Lx(x-xbar)."""
        L_inv = cholesky(Gammax, lower=True)
        c = solve_triangular(L_inv, x - xbar)
        return np.linalg.norm(c) ** 2

    # We can reduce the cost if Lx is diagonal
    def _cost_function(self, x, A, indexes: list[int]):
        Gammaex = self.cond_covariance(x, indexes)
        A = self.block_control_matrix([self.constraint_matrix(i) for i in indexes])
        xbar = np.concatenate(
            [self.current_mean, *[self.control_fields[i] for i in indexes[1:]]]
        )

        loss = self._cond_term(Gammaex, A, x)
        regularization = self._distribution_term(self.partial_cov(indexes), x, xbar)
        return loss, regularization

    @staticmethod
    def block_control_matrix(constraint_matrices: list):
        return bmat(
            [[constraint_matrices[0]] + [None] * (len(constraint_matrices) - 1)]
            + [
                [c] + [None] * i + [c] + [None] * (len(constraint_matrices) - i - 2)
                for i, c in enumerate(constraint_matrices[1:])
            ]
        )

    def minimization_problem(self, indexes: list[int]) -> dict:
        A = self.block_control_matrix([self.constraint_matrices[i] for i in indexes])

        def cost_function(x):
            cost = self._cost_function(x, A, indexes)
            return sum(cost)

        # x0 = np.concatenate([classical_learn_hamiltonian(self.states[0],3), *[self.control_fields[i] for i in range(1,len(self.control_fields))] ])
        x0 = np.concatenate(
            [self.current_mean, *[self.control_fields[i] for i in indexes[1:]]]
        )

        assert (
            x0.size == A.shape[1]
        ), f"Size of x0: {x0.size} and constraint matrix: {A.shape} don't match"

        print(
            x0.shape,
            A.shape,
        )

        def callback(xx):
            return self._cost_function(xx, A, indexes)

        return {"fun": cost_function, "x0": x0, "callback": callback}

    def update_mean(self):
        min_prob = self.minimization_problem()
        posterior_mean = minimize(
            **min_prob, options={"maxiter": 1e5, "xrtol": 1e-3, "disp": True}
        ).x
        print(
            f"The cost function ends up with a value of:{min_prob['fun'](posterior_mean) }, it started with a value of {min_prob['fun'](min_prob['x0'])}"
        )
        return posterior_mean

    def update_cov(self, posterior_x):
        A = self.block_control_matrix(self.constraint_matrices)
        Gammaex = self.cond_covariance(posterior_x)
        invgammaex = np.linalg.inv(Gammaex)
        return np.linalg.inv(self.inverse_cov + A.T @ invgammaex @ A)
