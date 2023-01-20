from dataclasses import dataclass
from typing import Optional, Union
import logging

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp

from .expectation import Expectation
from .fidelity import Fidelity
from .gradients import Gradient, FidelityGradient, ExpectationParameterShift, FidelityParameterShift


logger = logging.getLogger(__name__)


@dataclass
class GradientDescentOptions:
    _learning_rate: Union[float, np.ndarray] = 0.01

    @property
    def learning_rate(self):
        if isinstance(self._learning_rate, float):
            return self._learning_rate * np.ones(self.maxiter)
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    maxiter: int = 100
    losstol: float = 1e-5
    gradtol: float = 1e-3
    blocking: bool = False


@dataclass
class DualITEResult:
    times: list
    parameters: list
    energies: list
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
        expectation_gradient: Optional[Gradient] = None,
        fidelity_gradient: Optional[FidelityGradient] = None,
        norm_factor: float = 1,
        arccos: bool = False,
        gd_options: Optional[GradientDescentOptions] = None,
        warmstart: Union[bool, np.ndarray] = True,
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
            gd_options: Gradient descent options.
            warmstart: If True use the gradient of the last iteration as initial step.
                If a vector, this vector is used as first initial guess and then the other
                steps are warmstarted. If False a zero vector is used as initial guess.
            timestep_normalization: If True, normalize the learning rate by the timestep.
        """
        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.expectation = expectation
        self.fidelity = fidelity

        self.expectation.circuit = ansatz
        self.fidelity.right_circuit = ansatz
        self.fidelity.left_circuit = ansatz

        if expectation_gradient is None:
            expectation_gradient = ExpectationParameterShift()

        expectation_gradient.expectation = self.expectation
        self.expectation_gradient = expectation_gradient

        if fidelity_gradient is None:
            fidelity_gradient = FidelityParameterShift()

        fidelity_gradient.fidelity = self.fidelity
        self.fidelity_gradient = fidelity_gradient

        self.norm_factor = norm_factor
        self.arccos = arccos

        if gd_options is None:
            gd_options = GradientDescentOptions()

        self.gd_options = gd_options
        self.warmstart = warmstart
        self.timestep_normalization = timestep_normalization

    def get_loss_and_gradient(self, dt: float, theta: np.ndarray):
        energy_gradient = 0.5 * self.expectation_gradient.compute(theta)

        def loss_and_gradient(x, return_gradient=True):
            fid = self.fidelity.evaluate(theta, theta + x)

            if self.arccos:
                infid = np.arccos(np.sqrt(fid)) ** 2
            else:
                infid = 1 - fid

            x_norm = np.linalg.norm(x)
            penalty = self.norm_factor * x_norm**2
            loss = np.dot(x, energy_gradient) + 0.5 / dt * infid * (1 + penalty)

            if not return_gradient:
                return loss

            # compute gradient
            infid_derivative = -self.fidelity_gradient.compute(theta, theta + x)

            if self.arccos:
                if np.isclose(fid, 1):
                    infid_derivative = infid_derivative * np.sqrt(2)
                else:
                    infid_derivative = (
                        infid_derivative * np.arccos(np.sqrt(fid)) / np.sqrt(2 * fid * (1 - fid))
                    )

            d1 = infid_derivative * (1 + penalty)
            d2 = 2 * self.norm_factor * x * infid
            gradient = energy_gradient + 0.5 / dt * (d1 + d2)

            return loss, gradient

        return loss_and_gradient

    def step(self, dt: float, theta: np.ndarray, x0: Optional[np.ndarray] = None):
        """Perform a single step of the time evolution.

        Args:
            dt: The time step.
            theta: The current parameters.
            x0: The point at which to start the optimization (0 per default).

        Returns:
            The new parameters, the losses and gradient norms of the optimization.
        """
        logger.info("Starting dual optimization...")
        losses = []
        gradients = []
        thetas = []

        loss_and_gradient = self.get_loss_and_gradient(dt, theta)

        if not hasattr(self.gd_options.learning_rate, "__iter__"):
            maxiter = self.gd_options.maxiter
            learning_rate = self.gd_options.learning_rate
            eta = iter(np.ones(maxiter) * learning_rate)
        else:
            eta = iter(self.gd_options.learning_rate)

        if x0 is None:
            x = 0.01 * np.ones_like(theta)
        else:
            x = x0

        # if self.arccos:
        #     infid = np.arccos(np.sqrt(self.fidelity(theta, theta + x))) ** 2
        # else:
        #     infid = 1 - self.fidelity.evaluate(theta, theta + x)

        # loss = np.dot(x, energy_gradient) + 0.5 / dt * infid * (
        #     1 + self.norm_factor * np.linalg.norm(x) ** 2
        # )
        # losses.append(loss)
        # losses.append(loss_and_gradient(x0, return_gradient=False))

        for _ in range(self.gd_options.maxiter):
            # fid = self.fidelity.evaluate(theta, theta + x)
            # if self.arccos:
            #     infid = np.arccos(np.sqrt(fid)) ** 2
            # else:
            #     infid = 1 - fid

            # x_norm = np.linalg.norm(x)

            # # compute gradient
            # infid_derivative = infidelity_gradient(theta + x)

            # if self.arccos:
            #     if np.isclose(fid, 1):
            #         infid_derivative = infid_derivative * np.sqrt(2)
            #     else:
            #         infid_derivative = (
            #             infid_derivative * np.arccos(np.sqrt(fid)) / np.sqrt(2 * fid * (1 - fid))
            #         )

            # d1 = infid_derivative * (1 + self.norm_factor * x_norm**2)
            # d2 = 2 * self.norm_factor * x * infid
            # gradient = energy_gradient + 0.5 / dt * (d1 + d2)
            loss, gradient = loss_and_gradient(x)

            # update iterate
            x = x - next(eta) * gradient

            # loss = np.dot(x, energy_gradient) + 0.5 / dt * infid * (
            #     1 + self.norm_factor * x_norm**2
            # )

            # blocking criterion
            if self.gd_options.blocking and len(losses) > 1:
                if loss - losses[-1] > 0.01:
                    logger.info("-- Rejecting GD step.")
                    continue

            losses.append(loss)
            gradients.append(np.linalg.norm(gradient))
            thetas.append(theta + x)

            logger.info("-- Loss: %f, gradientnorm: %f", loss, np.linalg.norm(gradient))

            if len(losses) >= 2 and np.abs(losses[-1] - losses[-2]) < self.gd_options.losstol:
                print("-- Reached losstol.")
                break

            if np.linalg.norm(gradient) < self.gd_options.gradtol:
                print("-- Reached tolerance.")
                break

        return theta + x, losses, gradients

    def evolve(self, hamiltonian: PauliSumOp, final_time: float, timestep: float) -> DualITEResult:
        """Evolve the ansatz circuit for a given Hamiltonian.

        Args:
            hamiltonian: The Hamiltonian to evolve under.
            final_time: The final time.
            timestep: The timestep.

        Returns:
            The result of the evolution.
        """
        self.expectation.hamiltonian = hamiltonian
        energies = [self.expectation.evaluate(self.initial_parameters)]
        parameters = [self.initial_parameters]

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
            print("-- Warmstarting with initial guess:", x0)
            next_theta, losses_i, gradients_i = self.step(timestep, parameters[-1], x0)

            if self.warmstart is not False:  # to cover the case of np.ndarray
                x0 = next_theta - parameters[-1]

            parameters.append(next_theta)
            energies.append(self.expectation.evaluate(next_theta))
            losses.append(losses_i)
            gradients.append(gradients_i)

        return DualITEResult(times, parameters, energies, losses, gradients)
