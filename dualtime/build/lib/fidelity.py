"""Evaluate the fidelity of the ansatz for two parameter sets."""

from typing import Union, Optional, List
import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit

from qiskit.opflow import CircuitSampler, StateFn, ExpectationBase
from qiskit.providers import Backend

from .expectation import Expectation


class Fidelity:
    """Class to compute the fidelity of two quantum states represented by circuits."""

    def __init__(
        self,
        backend: Backend,
        expectation: ExpectationBase,
        left_circuit: Optional[QuantumCircuit] = None,
        right_circuit: Optional[QuantumCircuit] = None,
        shots: Optional[int] = None,
        measurement_error_mitigation: bool = False,
        initial_layout: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            backend: The backend to use.
            expectation: The expectation converter to use for the basis transformation.
            left_circuit: The first circuit to be applied.
            right_circuit: The second circuit (which will be inverted).
            shots: Number of shots for the backend.
            measurement_error_mitigation: If True use a complete measurement fitter to
                mitigate readout error.
            initial_layout: The initial layout to map the circuit qubits to physical qubits on
                the device.
        """
        self.left_circuit = left_circuit
        self.right_circuit = right_circuit
        self.expectation = expectation

        quantum_instance = Expectation._construct_quantum_instance(
            backend, shots, measurement_error_mitigation, initial_layout
        )

        self.sampler = CircuitSampler(quantum_instance)

        self.projection = None
        self.left_parameters = None
        self.right_parameters = None

    @property
    def left_circuit(self) -> QuantumCircuit:
        """The circuit preparing the quantum state."""
        return self._left_circuit

    @left_circuit.setter
    def left_circuit(self, circuit: QuantumCircuit) -> None:
        """Set the circuit preparing the quantum state."""
        self._left_circuit = circuit
        self.projection = None

    @property
    def right_circuit(self) -> QuantumCircuit:
        """The circuit preparing the quantum state."""
        return self._right_circuit

    @right_circuit.setter
    def right_circuit(self, circuit: QuantumCircuit) -> None:
        """Set the circuit preparing the quantum state."""
        self._right_circuit = circuit
        self.projection = None

    def set_projection(self):
        a = ParameterVector("a", self.left_circuit.num_parameters)
        b = ParameterVector("b", self.right_circuit.num_parameters)
        self.left_parameters = a
        self.right_parameters = b

        overlap = self.left_circuit.assign_parameters(a)
        overlap.compose(self.right_circuit.assign_parameters(b).inverse(), inplace=True)

        num_qubits = self.left_circuit.num_qubits
        projection = StateFn("0" * num_qubits, is_measurement=True) @ StateFn(overlap)
        self.projection = self.expectation.convert(projection)

    def evaluate(
        self,
        left_parameters: Union[np.ndarray, List[np.ndarray]],
        right_parameters: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[float, List[float]]:
        """Evaluate the fidelity.

        Args:
            left_parameters: Parameter values for the left circuit.
            right_parameters: Parameter values for the right circuit.

        Returns:
            The fidelity (or fidelities if a list of parameter values was supplied).

        Raises:
            ValueError: If the two parameter sets are incompatible.
        """
        if self.projection is None:
            self.set_projection()

        if isinstance(left_parameters, list) and isinstance(right_parameters, list):
            left_parameters = np.array(left_parameters)
            right_parameters = np.array(right_parameters)
            value_dict = {
                parameter: left_parameters[:, i] for i, parameter in enumerate(self.left_parameters)
            }
            value_dict.update(
                {
                    parameter: right_parameters[:, i]
                    for i, parameter in enumerate(self.right_parameters)
                }
            )
        elif isinstance(left_parameters, np.ndarray) and isinstance(right_parameters, np.ndarray):
            value_dict = dict(zip(self.left_parameters, left_parameters))
            value_dict.update(dict(zip(self.right_parameters, right_parameters)))
        else:
            raise ValueError(
                "Left and right parameters must be both a list or both a single array."
            )

        sampled = self.sampler.convert(self.projection, params=value_dict)
        amplitude = sampled.eval()
        return np.abs(amplitude) ** 2
