# pylint: disable=import-error, no-name-in-module

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import logging
import coloredlogs
from dataclasses import asdict

from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp, AerPauliExpectation, PauliExpectation
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

from src.expectation import Expectation
from src.fidelity import Fidelity

from src.gradients import ExpectationParameterShift, FidelityParameterShift
from src.gradients import ExpectationReverse, FidelityReverse
from src.dual import DualITE
from src.mclachlan import VarQITE
from src.optimizers import SPSA, GradientDescent

from src.qmetts import QMETTS

from experiments.models.heisenline import get_model
from experiments.qmetts.initial_states import random_basis

# coloredlogs.install(level="INFO", logger=logging.getLogger("src.dual"))

reps = 10
dirname = "data/heisen2/qmetts20_nostop"
# filenames = ["dump.npy"]
filenames = [f"{dirname}/{i + 1}.npy" for i in range(1, reps + 1)]
# filenames = [dirname]

hamiltonian, circuit, _ = get_model(num_sites=2, reps=1, J=0.25, g=-1)
print(hamiltonian)

print(circuit.draw())

inverse_temperature = 1
timestep = 0.01

mode = "sv"
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

# gd100 = GradientDescent(maxiter=100, learning_rate=0.1, blocking=False, losstol=1e-7, gradtol=1e-5)
# gd100 = GradientDescent(maxiter=100, learning_rate=0.1, blocking=False, losstol=1e-8, gradtol=1e-5)
gd100 = GradientDescent(maxiter=100, learning_rate=0.1, blocking=False, losstol=0, gradtol=0)
# gd10 = GradientDescent(maxiter=10, learning_rate=0.1, blocking=False, losstol=0, gradtol=0)
# optimizers = [gd100] + 100 * [gd10]
optimizers = gd100


initial_parameters = np.zeros(circuit.num_parameters)
evolver = DualITE(
    circuit,
    initial_parameters,
    expectation,
    fidelity,
    expectation_gradient,
    fidelity_gradient,
    norm_factor=0,
    optimizer=optimizers,
    warmstart=True,
    timestep_normalization=True,
)

qmetts = QMETTS(
    evolver, num_samples=20, initial_product_state=lambda: random_basis(hamiltonian.num_qubits)
)
observables = [hamiltonian]

for filename in filenames:
    result = qmetts.estimate(hamiltonian, inverse_temperature, observables)
    np.save(filename, asdict(result))
