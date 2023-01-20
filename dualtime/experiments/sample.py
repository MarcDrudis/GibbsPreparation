# pylint: disable=import-error, no-name-in-module

import set_paths  # pylint: disable=unused-import

import numpy as np
import matplotlib.pyplot as plt
import logging
import coloredlogs

from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp, AerPauliExpectation, PauliExpectation
from qiskit.providers.aer import AerSimulator

from src.expectation import Expectation
from src.fidelity import Fidelity

from src.gradients import ExpectationParameterShift, FidelityParameterShift
from src.dual import DualITE

coloredlogs.install(level="INFO", logger=logging.getLogger("src.dual"))

hamiltonian = PauliSumOp.from_list([("XI", 1), ("IX", 1), ("ZZ", 1)])
circuit = RealAmplitudes(2, reps=1)
final_time = 1
timestep = 0.01

mode = "shots"
if mode == "sv":
    backend = AerSimulator(method="statevector")
    expectation_converter = AerPauliExpectation()
    shots = None
else:
    backend = AerSimulator()
    expectation_converter = PauliExpectation()
    shots = 1024

expectation = Expectation(backend, expectation_converter, shots=shots)
expectation_gradient = ExpectationParameterShift()
fidelity = Fidelity(backend, expectation_converter, shots=shots)
fidelity_gradient = FidelityParameterShift()

initial_parameters = np.zeros(circuit.num_parameters)
dual = DualITE(
    circuit, initial_parameters, expectation, fidelity, expectation_gradient, fidelity_gradient
)
result = dual.evolve(hamiltonian, final_time, timestep)

plt.plot(result.times, result.energies)
plt.show()
