# pylint: disable=import-error, no-name-in-module

import set_paths  # pylint: disable=unused-import

import numpy as np
import logging
import coloredlogs
from dataclasses import asdict

from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp, AerPauliExpectation, PauliExpectation
from qiskit.providers.aer import AerSimulator

from src.expectation import Expectation
from src.fidelity import Fidelity

from src.gradients import ExpectationParameterShift, FidelityParameterShift
from src.gradients import ExpectationReverse, FidelityReverse
from src.dual import DualITE
from src.mclachlan import VarQITE

coloredlogs.install(level="INFO", logger=logging.getLogger("src.dual"))

reps = 2
dirname = "data/triangle/shots"
filenames = [f"{dirname}/{i}.npy" for i in range(1, reps + 1)]
# filenames = ["data/triangle/varqite.npy"]


magn = 1
interact = 1
hamiltonian = PauliSumOp.from_list(
    [
        ("XII", magn),
        ("IXI", magn),
        ("IIX", magn),
        ("ZZI", interact),
        ("IZZ", interact),
        ("ZIZ", interact),
    ]
)
circuit = RealAmplitudes(3, reps=1, entanglement="reverse_linear")

print(circuit.draw())

final_time = 1
timestep = 0.01

mode = "shots"
if mode == "sv":
    backend = AerSimulator(method="statevector")
    expectation_converter = AerPauliExpectation()
    shots = None
    expectation_gradient = ExpectationReverse()
    fidelity_gradient = FidelityReverse()
else:
    backend = AerSimulator()
    expectation_converter = PauliExpectation()
    shots = 1024
    expectation_gradient = ExpectationParameterShift()
    fidelity_gradient = FidelityParameterShift()

expectation = Expectation(backend, expectation_converter, shots=shots)
fidelity = Fidelity(backend, expectation_converter, shots=shots)

initial_parameters = np.zeros(circuit.num_parameters)

# algo = VarQITE(circuit, initial_parameters, backend)
algo = DualITE(
    circuit,
    initial_parameters,
    expectation,
    fidelity,
    expectation_gradient,
    fidelity_gradient,
    warmstart=False,
)

for filename in filenames:
    result = algo.evolve(hamiltonian, final_time, timestep)
    np.save(filename, asdict(result))
