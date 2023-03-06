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
        control_fields: list[np.ndarray],
        cmat_factory: ConstraintMatrixFactory,
        prior_mean: np.ndarray,
        prior_covariance: float,
        prior_cfield_std: float,
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
        self.size = prior_mean.size
        self.current_mean = prior_mean
        self.current_cov = np.diag(np.ones_like(prior_mean) * prior_covariance)
        self.cfield_std = prior_cfield_std
        self.cmats = [None] * len(self.control_fields)
        self.error_mats = [None] * len(self.control_fields)

    def constraint_matrix(self, index: int) -> np.ndarray:
        if self.cmats[index] is None:
            k, e = self.cmat_factory.create_cmat(self.states[index], shots=self.shots)
            self.cmats[index] = k
            self.error_mats[index] = e
        return self.cmats[index]

    def error_matrix(self, index: int) -> np.ndarray:
        if self.error_mats[index] is None:
            self.cmats[index], self.error_mats[index] = self.cmat_factory.create_cmat(
                self.states[index], shots=self.shots
            )

        return self.error_mats[index]

    @staticmethod
    def _constraint_term(
        c_guess: np.ndarray,
        v_guesses: list[np.ndarray],
        cmats: list[np.ndarray],
        emats: list[np.ndarray],
    ):
        epsilons = [cmat @ (c_guess + v) for cmat, v in zip(cmats, v_guesses)]
        gammaseps = [
            np.einsum(
                "kio,i,j,ljo -> kl",
                emat,
                (c_guess + v),
                (c_guess + v),
                emat,
                optimize=True,
            )
            for emat, v in zip(emats, v_guesses)
        ]
        choleskis = [cholesky(geps, lower=True) for geps in gammaseps]
        # Instead of inverting the cholesky matrix it is easier to solve a system of linear equations.
        vectors = [
            solve_triangular(chol, eps) for chol, eps in zip(choleskis, epsilons)
        ]
        return sum([np.linalg.norm(v) ** 2 for v in vectors])

    @staticmethod
    def _regularization_V1(
        c_guess: np.ndarray,
        v_guesses: list[np.ndarray],
        c_prior: list[np.ndarray],
        v_priors: list[np.ndarray],
        covmat: np.ndarray,
        v_std: float,
    ):
        """In this approach we will consider a general covariance matrix for the original hamiltonian
        we are trying to learn and a constant covariance for the control field.
        """
        c_term = solve_triangular(cholesky(covmat, lower=True), c_guess - c_prior)
        v_terms = [(vg - vp) for vg, vp in zip(v_guesses, v_priors)]
        return np.linalg.norm(c_term) ** 2 + 1 / v_std**2 * sum(
            [np.linalg.norm(v) ** 2 for v in v_terms]
        )

    @staticmethod
    def _regularization_V2(
        c_guess: np.ndarray,
        v_guesses: list[np.ndarray],
        c_prior: list[np.ndarray],
        v_priors: list[np.ndarray],
        covmat: np.ndarray,
        v_std: float,
    ):
        learn_term = np.linalg.norm(c_guess - c_prior) ** 2
        cfield_term = sum(
            np.linalg.norm(vg - vp) ** 2 for vg, vp in zip(v_guesses, v_priors)
        )
        return 1e4 * (learn_term + cfield_term / v_std**2)

    def terms_cost_function(self, indices: list[int], x):
        cmats = [self.constraint_matrix(i) for i in indices]
        emats = [self.error_matrix(i) for i in indices]
        c_prior = self.current_mean
        covmat = self.current_cov
        v_std = self.cfield_std
        v_priors = [self.control_fields[i] for i in indices]
        c_guess, *v_guesses = np.split(x, len(indices) + 1)
        return self._constraint_term(
            c_guess, v_guesses, cmats, emats
        ), self._regularization_V2(c_guess, v_guesses, c_prior, v_priors, covmat, v_std)

    def cost_function(self, indices: list[int]):
        cmats = [self.constraint_matrix(i) for i in indices]
        emats = [self.error_matrix(i) for i in indices]
        c_prior = self.current_mean
        covmat = self.current_cov
        v_std = self.cfield_std
        v_priors = [self.control_fields[i] for i in indices]

        def cost_fun(x):
            c_guess, *v_guesses = np.split(x, len(indices) + 1)
            return self._constraint_term(
                c_guess, v_guesses, cmats, emats
            ) + self._regularization_V2(
                c_guess, v_guesses, c_prior, v_priors, covmat, v_std
            )

        x0 = np.concatenate([c_prior] + v_priors)
        return cost_fun, x0

    # @property
    # def current_cov(self):
    #     """Covariance of c, what we are interested. This is only used to compute the blocks in the conditional covariance"""
    #     return self.total_cov[: self.size, : self.size]

    # @current_cov.setter
    # def current_cov(self, covariance=np.ndarray):
    #     """Covariance of c, what we are interested. This is only used to compute the blocks in the conditional covariance"""
    #     self.total_cov[: self.size, : self.size] = covariance

    # def cfield_cov(self, index: int):
    #     """Covariance of v, the control field. We update the knowledge we had on the control fields as well.
    #     This is only used to compute the blocks in the conditional covariance"""
    #     if index == 0:
    #         return np.zeros((self.size, self.size))
    #     return self.total_cov[
    #         self.size * index : self.size * (index + 1),
    #         self.size * index : self.size * (index + 1),
    #     ]

    # def get_partial_cov(self, indexes: list[int]):
    #     pcov = self.total_cov
    #     pcov = np.concatenate(
    #         [pcov[i * self.size : (i + 1) * self.size, :] for i in indexes],
    #         axis=0,
    #     )
    #     pcov = np.concatenate(
    #         [pcov[:, i * self.size : (i + 1) * self.size] for i in indexes],
    #         axis=1,
    #     )
    #     return pcov

    # def set_partial_cov(self, new_cov: np.ndarray, indexes: list[int]):
    #     for ii, i in enumerate(indexes):
    #         for jj, j in enumerate(indexes):
    #             self.total_cov[
    #                 self.size * i : self.size * (i + 1),
    #                 self.size * j : self.size * (j + 1),
    #             ] = new_cov[
    #                 self.size * ii : self.size * (ii + 1),
    #                 self.size * jj : self.size * (jj + 1),
    #             ]

    # def cond_covariance(self, x, indexes: list[int]) -> np.ndarray:
    #     """
    #     Returns the covariance of the error conditioned on x.
    #     """
    #     cov = block_diag(
    #         *[self._single_block_cond_cov(x, i, e) for e, i in enumerate(indexes)]
    #     )
    #     return cov

    # def _single_block_cond_cov(self, x, c_field_index, enum_index):
    #     """
    #     This returns a block of the conditional covariance. The shape should be the same as c.
    #     """
    #     cfield_cov = self.cfield_cov(c_field_index)
    #     c = x[: self.size]
    #     v = (
    #         x[self.size * enum_index : self.size * (enum_index + 1)]
    #         if c_field_index != 0
    #         else np.zeros_like(c)
    #     )
    #     cov = self.current_cov + cfield_cov + np.outer(c, c) + np.outer(v, v)
    #     np.fill_diagonal(
    #         cov,
    #         np.trace(self.current_cov)
    #         + np.trace(cfield_cov)
    #         + np.dot(c, c)
    #         + np.dot(v, v),
    #     )

    #     return self.sampling_std * cov

    # @staticmethod
    # def _cond_term(Gammaex, A, x):
    #     """Given Gammaex, A and x compute the term
    #     Lex@A@x , where Lex is the cholesky decompostion of Gammaex inverse.
    #     We use the fact that cholesky of the inverse is the inverse of cholesky.
    #     Be warry that scipy uses different notation than paper.
    #     """
    #     L_inv = cholesky(Gammaex, lower=True)
    #     c = solve_triangular(L_inv, A @ x)
    #     return np.linalg.norm(c) ** 2

    # @staticmethod
    # def _distribution_term(Gammax, x, xbar):
    #     """Returns the other term to minimize Lx(x-xbar)."""
    #     L_inv = cholesky(Gammax, lower=True)
    #     c = solve_triangular(L_inv, x - xbar)
    #     return np.linalg.norm(c) ** 2

    # # We can reduce the cost if Lx is diagonal
    # def _cost_function(self, x, A, indexes: list[int]):
    #     Gammaex = self.cond_covariance(x, indexes)
    #     A = self.block_control_matrix(indexes)

    #     xbar = np.concatenate(
    #         [self.current_mean, *[self.control_fields[i] for i in indexes[1:]]]
    #     )
    #     loss = self._cond_term(Gammaex, A, x)
    #     regularization = self._distribution_term(self.get_partial_cov(indexes), x, xbar)
    #     return loss, regularization

    # def block_control_matrix(self, indexes=list[int]):
    #     constraint_matrices = [self.constraint_matrix(i) for i in indexes]
    #     return bmat(
    #         [[constraint_matrices[0]] + [None] * (len(constraint_matrices) - 1)]
    #         + [
    #             [c] + [None] * i + [c] + [None] * (len(constraint_matrices) - i - 2)
    #             for i, c in enumerate(constraint_matrices[1:])
    #         ]
    #     )

    # def minimization_problem(self, indexes: list[int]) -> dict:
    #     A = self.block_control_matrix(indexes)

    #     def cost_function(x):
    #         cost = self._cost_function(x, A, indexes)
    #         return sum(cost)

    #     # x0 = np.concatenate([classical_learn_hamiltonian(self.states[0],3), *[self.control_fields[i] for i in range(1,len(self.control_fields))] ])
    #     x0 = np.concatenate(
    #         [self.current_mean, *[self.control_fields[i] for i in indexes[1:]]]
    #     )
    #     print(len(indexes), A.shape, x0.shape, self.current_mean.shape)
    #     assert (
    #         x0.size == A.shape[1]
    #     ), f"Size of x0: {x0.size} and constraint matrix: {A.shape} don't match"

    #     def callback(xx):
    #         return self._cost_function(xx, A, indexes)

    #     return {"fun": cost_function, "x0": x0, "callback": callback}

    # def update_mean(
    #     self,
    #     indexes: list[int],
    #     options: dict = {"maxiter": 1e5, "xrtol": 1e-3, "disp": True},
    # ):
    #     min_prob = self.minimization_problem(indexes)
    #     posterior_mean = minimize(**min_prob, options=options).x
    #     print(
    #         f"The cost function ends up with a value of:{min_prob['fun'](posterior_mean) }, it started with a value of {min_prob['fun'](min_prob['x0'])}"
    #     )
    #     assert (
    #         self.current_mean.shape == posterior_mean[: self.size].shape
    #     ), "Shapes don't match"
    #     self.current_mean = posterior_mean[: self.size]
    #     return posterior_mean

    # def update_cov(self, posterior_x, indexes):
    #     A = self.block_control_matrix(indexes)
    #     Gammaex = self.cond_covariance(posterior_x, indexes)
    #     invgammaex = np.linalg.inv(Gammaex)
    #     inverse_cov = np.linalg.inv(self.get_partial_cov(indexes))
    #     new_cov = np.linalg.inv(inverse_cov + A.T @ invgammaex @ A)
    #     self.set_partial_cov(new_cov, indexes)
    #     return new_cov
