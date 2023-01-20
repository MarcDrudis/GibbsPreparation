from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import scipy as sc

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import StatePreparation

from .dual import DualITE


class QMETTS:
    def __init__(
        self, evolver: DualITE, num_samples: int, initial_product_state: Callable[[], str]
    ) -> None:
        """
        Args:
            evolver: The imaginary time evolver.
            num_samples: The number of METTS samples.
            initial_product_state: A callable generating the initial product state.
        """
        self.evolver = evolver
        self.num_samples = num_samples
        self.initial_product_state = initial_product_state

    def estimate(
        self,
        hamiltonian: SparsePauliOp,
        inverse_temperature: float,
        observables: list[SparsePauliOp],
        timestep: float = 0.01,
    ):
        r"""Compute the ensemble average of the observables.

        Args:
            hamiltonian: The Hamiltonian of the system.
            inverse_temperature: The inverse temperature (or :math:`\beta`) of the system.
            observables: The observables whose ensemble average is approximated.
            timestep: The timestep for the forward Euler integration.

        Returns:
            A result object containing each initial product state, each observable sample,
            and their averages.
        """
        # create initial product state
        bitstring = self.initial_product_state()
        initial_state = StatePreparation(bitstring)

        # we append the initial state to the circuit and start from 0 parameters
        # alternatively, the last layer of the circuit could be tuned to prepare the initial state
        ansatz = self.evolver.ansatz
        self.evolver.initial_parameters = np.zeros(ansatz.num_parameters)

        hadamards = QuantumCircuit(ansatz.num_qubits)
        hadamards.h(hadamards.qubits)

        H = hamiltonian.primitive.to_matrix(sparse=True)
        U = sc.sparse.linalg.expm(-inverse_temperature / 2 * H)
        O = [obs.primitive.to_matrix(sparse=True) for obs in observables]

        samples = []
        reference_samples = []
        bitstrings = [bitstring]
        evolver_results = []

        for i in range(self.num_samples):
            # prepare ansatz with initial state
            model = ansatz.compose(initial_state, ansatz.qubits)
            self.evolver.ansatz = model

            print("Initial bitstring", bitstring)
            print("Model\n", model.decompose().draw())

            # evolve
            qite_result = self.evolver.evolve(hamiltonian, inverse_temperature / 2, timestep)
            evolver_results.append(qite_result)
            final_parameters = qite_result.parameters[-1]
            print("Energies:", qite_result.energies)

            # sample observables
            # samples.append(self._sample_observables(model, final_parameters, observables))
            samples.append(self._sample_observables(model, final_parameters, observables))
            print("Observables:", samples[-1])

            # reference values
            exact_state = U.dot(Statevector.from_label(bitstring))
            norm = np.conj(exact_state).dot(exact_state)
            reference_samples_ = [
                np.conj(exact_state).dot(obs.dot(exact_state)) / norm for obs in O
            ]
            print("Reference:", reference_samples_)
            reference_samples.append(reference_samples_)

            # stop here if no more samples are drawn
            if i == self.num_samples - 2:
                break

            # switch basis
            if i % 2 == 0:
                model.compose(hadamards, model.qubits, inplace=True)

            # sample new product state
            bound = model.bind_parameters(final_parameters)
            bound.measure_all()
            bitstring = self._sample_bitstring(bound)

            if i % 2 == 0:
                bitstring = "".join(map(lambda bit: "+" if bit == "0" else "-", bitstring))

            bitstrings.append(bitstring)
            initial_state = StatePreparation(bitstring)

        # reset ansatz of the evolver
        self.evolver.ansatz = ansatz

        # compute averages and create result object
        averages = np.mean(samples, axis=0)
        stds = np.std(samples, axis=0)

        result = QMETTSResult(
            averages, stds, samples, bitstrings, evolver_results, reference_samples
        )
        return result

    def _sample_observables_exact(self, model, parameters, observables):
        results = [
            Statevector(model.bind_parameters(parameters)).expectation_value(obs)
            for obs in observables
        ]
        return results

    def _sample_observables(self, model, parameters, observables):
        expectation = self.evolver.expectation
        expectation.circuit = model
        results = []
        for obs in observables:
            expectation.hamiltonian = obs
            results.append(expectation.evaluate(parameters))

        return results

    def _sample_bitstring(self, model):
        backend = self.evolver.expectation.backend
        transpiled = transpile(model, backend)
        result = backend.run(transpiled, shots=1, memory=True).result()
        return result.get_memory()[0]


@dataclass
class QMETTSResult:
    averages: list
    stds: list
    samples: list
    bitstrings: list
    evolver_results: list
    reference_samples: list
