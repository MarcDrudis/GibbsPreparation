"""Evaluate the expectation of a Hamiltonian given an ansatz and a parameter set."""

import numpy as np
from typing import Optional, List, Union

from qiskit.circuit import QuantumCircuit

from qiskit.opflow import StateFn, CircuitSampler, PauliSumOp, ExpectationBase
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance

from qiskit.ignis.mitigation.measurement import CompleteMeasFitter


class Expectation:
    """An ``Expectation`` primitive."""

    def __init__(
        self,
        backend: Optional[Backend],
        expectation: Optional[ExpectationBase],
        hamiltonian: Optional[PauliSumOp] = None,
        circuit: Optional[QuantumCircuit] = None,
        shots: Optional[int] = None,
        measurement_error_mitigation: bool = False,
        initial_layout: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            backend: The backend to use.
            expectation: The expectation converter to use for the basis transformation.
            hamiltonian: The Hamiltonian for which to evaluate the expectation.
            circuit: The circuit preparing the quantum state.
            shots: Number of shots for the backend.
            measurement_error_mitigation: If True use a complete measurement fitter to
                mitigate readout error.
            initial_layout: The initial layout to map the circuit qubits to physical qubits on
                the device.
        """
        self._hamiltonian = None
        if hamiltonian is not None:
            self.hamiltonian = hamiltonian

        self._circuit = None
        if circuit is not None:
            self.circuit = circuit

        self.backend = backend
        self.expectation = expectation

        quantum_instance = self._construct_quantum_instance(
            backend, shots, measurement_error_mitigation, initial_layout
        )
        self.quantum_instance = quantum_instance
        self.sampler = CircuitSampler(quantum_instance)

        self.exp = None

    def set_expectation(self):
        """Set the expectation based on circuit and hamiltonian."""
        required_attrs = {"circuit", "hamiltonian"}
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError(f"{attr} is required.")

        exp = StateFn(self.hamiltonian, is_measurement=True).compose(StateFn(self.circuit))
        self.exp = self.expectation.convert(exp)

    @property
    def hamiltonian(self) -> PauliSumOp:
        """The Hamiltonian for which to evaluate the expectation."""
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian: PauliSumOp) -> None:
        """Set the Hamiltonian for which to evaluate the expectation."""
        self._hamiltonian = hamiltonian
        self.exp = None

    @property
    def circuit(self) -> QuantumCircuit:
        """The circuit preparing the quantum state."""
        return self._circuit

    @circuit.setter
    def circuit(self, circuit: QuantumCircuit) -> None:
        """Set the circuit preparing the quantum state."""
        self._circuit = circuit
        self.parameters = circuit.parameters
        self.exp = None

    def evaluate(
        self, parameters: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[float, List[float]]:
        """Evaluate the expectation value.

        Args:
            parameters: The parameter values for the quantum state.

        Returns:
            The expectation(s).
        """
        if self.exp is None:
            self.set_expectation()

        if isinstance(parameters, list):
            parameters = np.array(parameters)
            value_dict = {
                parameter: parameters[:, i].tolist() for i, parameter in enumerate(self.parameters)
            }
        else:
            value_dict = dict(zip(self.parameters, parameters))

        sampled = self.sampler.convert(self.exp, params=value_dict)
        return np.real(sampled.eval())

    @staticmethod
    def _construct_quantum_instance(backend, shots, measurement_error_mitigation, initial_layout):
        if measurement_error_mitigation:
            quantum_instance = QuantumInstance(
                backend,
                shots=shots,
                initial_layout=initial_layout,
                measurement_error_mitigation_shots=shots,
                measurement_error_mitigation_cls=CompleteMeasFitter,
            )
        else:
            quantum_instance = QuantumInstance(backend, initial_layout=initial_layout, shots=shots)

        return quantum_instance
