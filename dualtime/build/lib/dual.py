from __future__ import annotations
from dataclasses import dataclass
import logging

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp

from .expectation import Expectation
from .fidelity import Fidelity
from .gradients import Gradient, FidelityGradient, ExpectationParameterShift, FidelityParameterShift
from .optimizers.gd import GradientDescent


logger = logging.getLogger(__name__)


@dataclass
class DualITEResult:
    times: list
    parameters: list
    energies: list

    # history attributes
    thetas: list
    losses: list
    gradients: list


class DualITE:
    """Imaginary time evolution using the QNG minimization formulation."""

    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_parameters: np.ndarray,
        expectation: Expectation,
        fidelity: Fidelity,
        expectation_gradient: Gradient | None = None,
        fidelity_gradient: FidelityGradient | None = None,
        norm_factor: float = 1,
        arccos: bool = False,
        optimizer: callable | list[callable] | None = None,
        warmstart: bool | np.ndarray = True,
        timestep_normalization: bool = False,
    ):
        """
        Args:
            ansatz: The ansatz circuit.
            expectation: The expectation converter to use for the basis transformation.
            fidelity: The fidelity of the two quantum states.
            expectation_gradient: The expectation gradient.
            fidelity_gradient: The fidelity gradient.
            norm_factor: Add the norm of the update step to the loss with this factor.
            arccos: Use arccos instead of 1 - fidelity.
            optimizer: An optimizer callable, taking a loss_and_gradient and initial point.
            warmstart: If True use the gradient of the last iteration as initial step.
                If a vector, this vector is used as first initial guess and then the other
                steps are warmstarted. If False a zero vector is used as initial guess.
            timestep_normalization: If True, normalize the learning rate by the timestep.
        """
        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.expectation = expectation
        self.fidelity = fidelity

        if expectation_gradient is None:
            expectation_gradient = ExpectationParameterShift()

        self.expectation_gradient = expectation_gradient

        if fidelity_gradient is None:
            fidelity_gradient = FidelityParameterShift()

        self.fidelity_gradient = fidelity_gradient

        self.norm_factor = norm_factor
        self.arccos = arccos

        if optimizer is None:
            optimizer = GradientDescent(maxiter=300, learning_rate=0.01)

        self.optimizer = optimizer

        self.warmstart = warmstart
        self.timestep_normalization = timestep_normalization

    def get_loss_and_gradient(self, dt: float, theta: np.ndarray):
        energy_gradient = 0.5 * self.expectation_gradient.compute(theta)

        def loss_and_gradient(x, return_gradient=True):
            # cast x to list of arrays
            return_list = True
            if len(np.asarray(x).shape) == 1:
                x = [x]
                return_list = False
            elif isinstance(x, np.ndarray):  # cast to list of array
                x = x.tolist()

            n = len(x)  # number of evaluations
            shifted = [theta + x_i for x_i in x]

            fids = self.fidelity.evaluate(n * [theta], shifted)
            infids = 1 - fids

            x_norms = np.array([np.linalg.norm(x_i) for x_i in x])
            penalty = self.norm_factor * x_norms**2
            losses = [
                np.dot(x_i, energy_gradient) + 0.5 / dt * infids[i] * (1 + penalty[i])
                for i, x_i in enumerate(x)
            ]

            if self.timestep_normalization:
                losses = [loss * dt for loss in losses]

            if not return_gradient:
                return losses if return_list else losses[0]

            if len(x) > 1:
                raise RuntimeError("What do you need multiple gradients for?")

            # compute gradient
            fid_derivatives = self.fidelity_gradient.compute(n * [theta], shifted)

            d1 = [-fid_derivatives[i] * (1 + penalty[i]) for i in range(n)]
            d2 = [2 * self.norm_factor * x_i * infids[i] for i, x_i in enumerate(x)]
            gradients = [energy_gradient + 0.5 / dt * (d1[i] + d2[i]) for i in range(n)]

            if self.timestep_normalization:
                gradients = [gradient * dt for gradient in gradients]

            if return_list:
                return losses, gradients
            return losses[0], gradients[0]

        return loss_and_gradient

    def step(
        self,
        dt: float,
        theta: np.ndarray,
        x0: np.ndarray | None = None,
        iteration: int | None = None,
    ):
        """Perform a single step of the time evolution.

        Args:
            dt: The time step.
            theta: The current parameters.
            x0: The point at which to start the optimization (0 per default).
            iteration: The iteration count. If a list of optimizers is passed, this chooses
                the ``iteration``-th element of that list for the optimization.

        Returns:
            The new parameters, along with: all parameters, all losses and all gradient norms of
            the optimization.
        """
        logger.info("Starting dual optimization...")

        loss_and_gradient = self.get_loss_and_gradient(dt, theta)

        if x0 is None:
            x = 0.01 * np.ones_like(theta)
        else:
            x = x0

        if isinstance(self.optimizer, list):
            if iteration is None:
                raise ValueError("If optimizer is a list, ``iteration`` must be specified.")
            if not iteration < len(self.optimizer):
                print("Warning! Iteration exceeded amount of optimizers, using the last one.")
                iteration = len(self.optimizer) - 1

            optimizer = self.optimizer[iteration]
        else:
            optimizer = self.optimizer

        x, xs, losses, gradients = optimizer(loss_and_gradient, x0)

        thetas = [theta + x_i for x_i in xs]
        return theta + x, thetas, losses, gradients

    def evolve(self, hamiltonian: PauliSumOp, final_time: float, timestep: float) -> DualITEResult:
        """Evolve the ansatz circuit for a given Hamiltonian.

        Args:
            hamiltonian: The Hamiltonian to evolve under.
            final_time: The final time.
            timestep: The timestep.

        Returns:
            The result of the evolution.
        """
        self._sync_primitives(hamiltonian)
        energies = [self.expectation.evaluate(self.initial_parameters)]
        parameters = [self.initial_parameters]

        thetas = []
        losses = []
        gradients = []

        num_timesteps = int(np.ceil(final_time / timestep))
        times = np.linspace(0, final_time, num_timesteps + 1).tolist()

        if isinstance(self.warmstart, np.ndarray):
            x0 = self.warmstart
        else:
            x0 = 0.01 * np.ones(self.initial_parameters.size)

        for i in range(num_timesteps):
            logger.info("Time %s/%s", times[i], final_time)
            # print("-- Warmstarting with initial guess:", x0)
            next_theta, thetas_i, losses_i, gradients_i = self.step(
                timestep, parameters[-1], x0, iteration=i
            )

            if self.warmstart is not False:  # to cover the case of np.ndarray
                x0 = next_theta - parameters[-1]

            parameters.append(next_theta)
            energies.append(self.expectation.evaluate(next_theta))
            logger.info("Energy %s", energies[-1])

            thetas.append(thetas_i)
            losses.append(losses_i)
            gradients.append(gradients_i)

        return DualITEResult(times, parameters, energies, thetas, losses, gradients)

    def _sync_primitives(self, hamiltonian):
        self.expectation.circuit = self.ansatz
        self.expectation.hamiltonian = hamiltonian

        self.fidelity.right_circuit = self.ansatz
        self.fidelity.left_circuit = self.ansatz

        self.expectation_gradient.expectation = self.expectation
        self.fidelity_gradient.fidelity = self.fidelity
