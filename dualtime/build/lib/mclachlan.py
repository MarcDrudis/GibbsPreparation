"""Exact imaginary time evolution based on matrix exponentiation."""

from typing import Optional, Callable
import numpy as np
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import (
    OperatorBase,
    ExpectationFactory,
    StateFn,
    CircuitSampler,
    QFI,
    Gradient,
    PauliSumOp,
)
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector

from surfer.qfi import ReverseQFI
from surfer.gradient import ReverseGradient

from scipy.integrate import odeint,quad,RK45


# ACTIVATE_TURBO = False


@dataclass
class VarQITEResult:
    """VarQITE result."""

    times: list
    parameters: list
    energies: list


class VarQITE:
    """Exact imaginary time evolution based on matrix exponentiation."""

    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_parameters: np.ndarray,
        backend: Backend,
        regularization: float = 0.01,
        lse_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        turbo_noise: None|float = None,
        activate_turbo: bool = False,
    ) -> None:
        """
        Args:
            ansatz: The parameterized circuit used for the variational time evolution.
            initial_parameters: The initial parameters for the circuit.
            backend: The backend to run the circuits on.
            perturbation: The small perturbation to estimate the gradients at the current parameters.
        """
        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.backend = backend
        self.regularization = regularization
        self.turbo_noise = turbo_noise
        self.activate_turbo = activate_turbo

        if lse_solver is None:
            lse_solver = np.linalg.solve

        self.lse_solver = lse_solver

    def evolve(
        self, hamiltonian: OperatorBase, final_time: float, timestep: float
    ) -> VarQITEResult:
        """TODO"""
        if self.activate_turbo:
            if not isinstance(hamiltonian, PauliSumOp):
                raise ValueError("Hamiltonian must be a PauliSumOp.")
            hamiltonian = hamiltonian.primitive.to_matrix(sparse=True)

        energy = self._get_energy_evaluation(hamiltonian)
        qgt = self._get_qgt_evaluation()
        b = self._get_b_evaluation(hamiltonian)

        times = [0]
        energies = [energy(self.initial_parameters)]
        x = [self.initial_parameters]
        qgts = []
        bs = []
        func = lambda x: self._evaluate_update(x,qgt,b)

        
        while times[-1]+10e-10 < final_time:
            print(times[-1])
            qgts.append(qgt(x[-1]))
            bs.append(b(x[-1]))
            
            # #Here goes the integration scheme.
            #Runge-Kutta 4th order
            k1 = func(x[-1])
            k2 = func(x[-1] + timestep/2 * k1)
            k3 = func(x[-1] + timestep/2 * k2)
            k4 = func(x[-1] + timestep * k3)
            slope = (k1 + 2*k2 + 2*k3 + k4)/6
            
            # #Euler
            # slope = func(x[-1])
            
            x.append(x[-1] + timestep * slope )
                    
            energies.append(energy(x[-1]))
            times.append(times[-1] + timestep)
        

        # build the final result object
        result = VarQITEResult(times, x, energies)
        return result
    
    def _evaluate_update(self,x:np.ndarray,qgt:Callable,b:Callable) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        qgt_instance = qgt(x)
        b_instance = b(x)
        parameter_derivative =  self.lse_solver(
                qgt_instance + self.regularization * np.identity(x.size), b_instance
            ).real
        return parameter_derivative

    def _get_energy_evaluation(self, hamiltonian):
        if self.activate_turbo:

            def evaluate_energy(parameters):
                state = Statevector(self.ansatz.bind_parameters(parameters)).data
                energy = np.conj(state.T).dot(hamiltonian.dot(state))
                if np.abs(np.imag(energy)) > 1e-10:
                    raise RuntimeError(f"Energy is not real: {energy}")
                return np.real(energy)

        else:
            expectation_converter = ExpectationFactory().build(hamiltonian, self.backend)
            energy = StateFn(hamiltonian, is_measurement=True).compose(StateFn(self.ansatz))
            energy = expectation_converter.convert(energy)
            sampler = CircuitSampler(self.backend)

            circuit_parameters = self.ansatz.parameters

            def evaluate_energy(parameters):
                param_dict = dict(zip(circuit_parameters, parameters))
                sampled = sampler.convert(energy, param_dict)
                return sampled.eval().real

        return evaluate_energy

    def _get_qgt_evaluation(self):
        if self.activate_turbo:
            qfi = ReverseQFI(do_checks=False)

            def evaluate_qgt(parameters):
                return qfi.compute(self.ansatz, parameters) / 4

        else:
            qfi = QFI().convert(StateFn(self.ansatz))
            # expectation_converter = ExpectationFactory().build(qfi, self.backend)
            # qfi = expectation_converter.convert(qfi)
            sampler = CircuitSampler(self.backend)
            ansatz_parameters = self.ansatz.parameters

            def evaluate_qgt(parameters):
                param_dict = dict(zip(ansatz_parameters, parameters))
                sampled = sampler.convert(qfi, param_dict)
                return sampled.eval() / 4

        return evaluate_qgt

    def _get_b_evaluation(self, hamiltonian):
        if self.activate_turbo:
            gradient = ReverseGradient(do_checks=False)

            def evaluate_b(parameters):
                return -np.real(gradient.compute(hamiltonian, self.ansatz, parameters)) / 2

        else:
            gradient = Gradient().convert(
                StateFn(hamiltonian, is_measurement=True).compose(StateFn(self.ansatz))
            )
            # expectation_converter = ExpectationFactory().build(gradient, self.backend)
            # gradient = expectation_converter.convert(gradient)
            sampler = CircuitSampler(self.backend)

            ansatz_parameters = self.ansatz.parameters

            def evaluate_b(parameters):
                param_dict = dict(zip(ansatz_parameters, parameters))
                sampled = sampler.convert(gradient, param_dict)
                return -np.asarray(sampled.eval()) / 2

        return evaluate_b
