# pylint: disable=import-error, no-name-in-module

import set_paths  # pylint: disable=unused-import

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

from models.heisenline import get_model

coloredlogs.install(level="INFO", logger=logging.getLogger("src.dual"))

reps = 5
# dirname = "data/heisen2/lrel1e1/sv/--_gd100.npy"
dirname = "dump.npy"
filenames = [dirname]
# filenames = [f"{dirname}/{i}.npy" for i in range(1, reps + 1)]
# filenames = ["data/heisen3/lr1e3/spsa_it100_b1/sv_xi1.npy"]

hamiltonian, circuit, initial_parameters = get_model(num_sites=2, reps=1, J=0.25, g=-1)
initial_parameters = np.zeros(circuit.num_parameters)
# circuit.x(0)
# circuit.x(1)
# circuit.h([0, 1])


print(circuit.draw())

final_time = 0.5
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


opt = GradientDescent(maxiter=100, learning_rate=0.1, blocking=False, losstol=0, gradtol=0)
# opt = SPSA(maxiter=100, learning_rate=0.1, perturbation=0.1, batch_size=1, averaging=1)


# algo = VarQITE(circuit, initial_parameters, backend)
algo = DualITE(
    circuit,
    initial_parameters,
    expectation,
    fidelity,
    expectation_gradient,
    fidelity_gradient,
    norm_factor=0,
    optimizer=opt,
    warmstart=True,
    timestep_normalization=True,
)

for filename in filenames:
    result = algo.evolve(hamiltonian, final_time, timestep)
    print(result.energies[-1])
    np.save(filename, asdict(result))
