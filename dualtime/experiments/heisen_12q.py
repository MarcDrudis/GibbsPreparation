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

reps = 4
dirname = "data/heisencomb/lrel1e1/shots1024/gd100_ws_nostop"
filenames = [f"{dirname}/{i}.npy" for i in range(1, reps + 1)]
# filenames = [dirname]

hamiltonian, circuit, initial_parameters = get_model(
    num_sites=12, reps=3, J=0.25, g=-1, periodic=True
)


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
    # plt.figure()
    # plt.plot(result.energies)
    # plt.figure()
    # plt.title("losses")
    # all_losses = []
    # for loss in result.losses:
    #     all_losses += loss
    # plt.plot(all_losses)
    # plt.show()
    np.save(filename, asdict(result))
