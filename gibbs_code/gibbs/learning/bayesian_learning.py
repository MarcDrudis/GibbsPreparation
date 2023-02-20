from __future__ import annotations

import time

import numpy as np
from gibbs.learning.constraint_matrix import ConstraintMatrixFactory
from gibbs.utils import classical_learn_hamiltonian, printarray
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from scipy.linalg import block_diag, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.sparse import bmat


class BayesianLearning:
    def __init__(
        self,
        states: list[Statevector | QuantumCircuit],
        control_fields: list[np.ndarrays],
        cmat_factory: ConstraintMatrixFactory,
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
        self.cmat_factory = cmat_factory
        self.sampling_std = sampling_std
        self.size = prior_mean.size
        self.current_mean = prior_mean
        if isinstance(prior_covariance, tuple):
            ones = np.ones(self.size)
            self.total_cov = np.diag(
                np.concatenate(
                    [ones * prior_covariance[0]]
                    + [ones * prior_covariance[1]] * (len(control_fields) - 1)
                )
            )
        else:
            self.total_cov = prior_covariance

        self.cmats = [None] * len(self.control_fields)

    def constraint_matrix(self, index: int) -> np.ndarray:
        if self.cmats[index] == None:
            self.cmats[index] = self.cmat_factory.create_cmat(
                self.states[index], shots=self.shots
            )

        return self.cmats[index]

    @property
    def current_cov(self):
        """Covariance of c, what we are interested. This is only used to compute the blocks in the conditional covariance"""
        return self.total_cov[: self.size, : self.size]

    @current_cov.setter
    def current_cov(self, covariance=np.ndarray):
        """Covariance of c, what we are interested. This is only used to compute the blocks in the conditional covariance"""
        self.total_cov[: self.size, : self.size] = covariance

    def cfield_cov(self, index: int):
        """Covariance of v, the control field. We update the knowledge we had on the control fields as well.
        This is only used to compute the blocks in the conditional covariance"""
        if index == 0:
            return np.zeros((self.size, self.size))
        return self.total_cov[
            self.size * index : self.size * (index + 1),
            self.size * index : self.size * (index + 1),
        ]

    def get_partial_cov(self, indexes: list[int]):
        pcov = self.total_cov
        pcov = np.concatenate(
            [pcov[i * self.size : (i + 1) * self.size, :] for i in indexes],
            axis=0,
        )
        pcov = np.concatenate(
            [pcov[:, i * self.size : (i + 1) * self.size] for i in indexes],
            axis=1,
        )
        return pcov

    def set_partial_cov(self, new_cov: np.ndarray, indexes: list[int]):
        for ii, i in enumerate(indexes):
            for jj, j in enumerate(indexes):
                self.total_cov[
                    self.size * i : self.size * (i + 1),
                    self.size * j : self.size * (j + 1),
                ] = new_cov[
                    self.size * ii : self.size * (ii + 1),
                    self.size * jj : self.size * (jj + 1),
                ]

    def cond_covariance(self, x, indexes: list[int]) -> np.ndarray:
        """
        Returns the covariance of the error conditioned on x.
        """
        cov = block_diag(
            *[self._single_block_cond_cov(x, i, e) for e, i in enumerate(indexes)]
        )
        return cov

    def _single_block_cond_cov(self, x, c_field_index, enum_index):
        """
        This returns a block of the conditional covariance. The shape should be the same as c.
        """
        cfield_cov = self.cfield_cov(c_field_index)
        c = x[: self.size]
        v = (
            x[self.size * enum_index : self.size * (enum_index + 1)]
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
        A = self.block_control_matrix(indexes)

        xbar = np.concatenate(
            [self.current_mean, *[self.control_fields[i] for i in indexes[1:]]]
        )
        loss = self._cond_term(Gammaex, A, x)
        regularization = self._distribution_term(self.get_partial_cov(indexes), x, xbar)
        return loss, regularization

    def block_control_matrix(self, indexes=list[int]):
        constraint_matrices = [self.constraint_matrix(i) for i in indexes]
        return bmat(
            [[constraint_matrices[0]] + [None] * (len(constraint_matrices) - 1)]
            + [
                [c] + [None] * i + [c] + [None] * (len(constraint_matrices) - i - 2)
                for i, c in enumerate(constraint_matrices[1:])
            ]
        )

    def minimization_problem(self, indexes: list[int]) -> dict:
        A = self.block_control_matrix(indexes)

        def cost_function(x):
            cost = self._cost_function(x, A, indexes)
            return sum(cost)

        # x0 = np.concatenate([classical_learn_hamiltonian(self.states[0],3), *[self.control_fields[i] for i in range(1,len(self.control_fields))] ])
        x0 = np.concatenate(
            [self.current_mean, *[self.control_fields[i] for i in indexes[1:]]]
        )
        print(len(indexes), A.shape, x0.shape, self.current_mean.shape)
        assert (
            x0.size == A.shape[1]
        ), f"Size of x0: {x0.size} and constraint matrix: {A.shape} don't match"

        def callback(xx):
            return self._cost_function(xx, A, indexes)

        return {"fun": cost_function, "x0": x0, "callback": callback}

    def update_mean(
        self,
        indexes: list[int],
        options: dict = {"maxiter": 1e5, "xrtol": 1e-3, "disp": True},
    ):
        min_prob = self.minimization_problem(indexes)
        posterior_mean = minimize(**min_prob, options=options).x
        print(
            f"The cost function ends up with a value of:{min_prob['fun'](posterior_mean) }, it started with a value of {min_prob['fun'](min_prob['x0'])}"
        )
        assert (
            self.current_mean.shape == posterior_mean[: self.size].shape
        ), "Shapes don't match"
        self.current_mean = posterior_mean[: self.size]
        return posterior_mean

    def update_cov(self, posterior_x, indexes):
        A = self.block_control_matrix(indexes)
        Gammaex = self.cond_covariance(posterior_x, indexes)
        invgammaex = np.linalg.inv(Gammaex)
        inverse_cov = np.linalg.inv(self.get_partial_cov(indexes))
        new_cov = np.linalg.inv(inverse_cov + A.T @ invgammaex @ A)
        self.set_partial_cov(new_cov, indexes)
        return new_cov
