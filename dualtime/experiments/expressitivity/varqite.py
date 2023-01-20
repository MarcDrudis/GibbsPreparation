"""Exact VarQITE until convergence."""

from collections.abc import Callable
import numpy as np
import scipy as sc

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliSumOp

from surfer.qfi import ReverseQFI
from surfer.gradient import ReverseGradient


class ConvQite:
    """Exact VarQITE until convergence."""

    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_parameters: np.ndarray,
        tol: float = 1e-4,
        max_time: float = 5,
        timestep: float = 0.01,
        regularization: float = 0.01,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """
        Args:
            ansatz: The parameterized circuit used for the variational time evolution.
            initial_parameters: The initial parameters for the circuit.
            tol: Absolute tolerance of how close to the ground state we must be.
            max_time: The maximum time we propagate.
            timestep: The timestep.
            perturbation: The small perturbation to estimate the gradients at the current parameters.
        """
        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.tol = tol
        self.max_time = max_time
        self.timestep = timestep
        self.regularization = regularization

        if lse_solver is None:
            lse_solver = np.linalg.solve

        self.lse_solver = lse_solver

    def quantum_evolve(self, hamiltonian: PauliSumOp, reference: str | dict) -> dict:
        if isinstance(hamiltonian, PauliSumOp):
            hamiltonian = hamiltonian.primitive.to_matrix(sparse=True)
        # hamiltonian = hamiltonian.primitive.to_matrix()

        if isinstance(reference, str):
            reference = np.load(reference, allow_pickle=True).item()

        times = reference["times"]
        ref_energies = reference["energies"]
        target_energy = reference["target_energy"]

        energy = self._get_energy_evaluation(hamiltonian)
        qgt = self._get_qgt_evaluation()
        b = self._get_b_evaluation(hamiltonian)

        energies = [energy(self.initial_parameters)]
        theta = self.initial_parameters
        iden = np.identity(theta.size)
        total_error = 0

        for time, ref_energy in zip(times, ref_energies):
            update = self.lse_solver(qgt(theta) + self.regularization * iden, b(theta)).real
            theta = theta + self.timestep * update
            energies.append(energy(theta))
            error = np.abs(energies[-1] - ref_energy)
            total_error += error
            print(time, energies[-1], error, end="\r")

        print("\nAverage error:", total_error / len(times))
        return {
            "times": times,
            "energies": energies,
        }

    def classical_reference(self, hamiltonian):
        hamiltonian = hamiltonian.primitive.to_matrix(sparse=True)
        target_energy = sc.sparse.linalg.eigsh(hamiltonian, k=1, which="SA")[0]
        state = Statevector(self.ansatz.bind_parameters(self.initial_parameters)).data
        print("Target energy", target_energy)

        times = [0]
        energies = [np.dot(np.conj(state), hamiltonian.dot(state))]

        while times[-1] < self.max_time:
            times.append(times[-1] + self.timestep)
            statevector = sc.sparse.linalg.expm(-times[-1] * hamiltonian).dot(state)
            # statevector = sc.sparse.linalg.expm_multiply(-times[-1] * hamiltonian, state.data)
            statevector /= np.linalg.norm(statevector)

            energy = np.conj(statevector).T.dot(hamiltonian.dot(statevector))
            energies.append(energy)
            print(times[-1], energies[-1], end="\r")

            if np.abs(energies[-1] - target_energy) < self.tol:
                print(f"\nReached convergence at t = {times[-1]}")
                break

        return hamiltonian, {"times": times, "energies": energies, "target_energy": target_energy}

    def _get_energy_evaluation(self, hamiltonian):
        def evaluate_energy(parameters):
            state = Statevector(self.ansatz.bind_parameters(parameters)).data
            energy = np.conj(state.T).dot(hamiltonian.dot(state))
            if np.abs(np.imag(energy)) > 1e-10:
                raise RuntimeError(f"Energy is not real: {energy}")
            return np.real(energy)

        return evaluate_energy

    def _get_qgt_evaluation(self):
        qfi = ReverseQFI(do_checks=False)

        def evaluate_qgt(parameters):
            return qfi.compute(self.ansatz, parameters) / 4

        return evaluate_qgt

    def _get_b_evaluation(self, hamiltonian):
        gradient = ReverseGradient(do_checks=False)

        def evaluate_b(parameters):
            return -np.real(gradient.compute(hamiltonian, self.ansatz, parameters)) / 2

        return evaluate_b
