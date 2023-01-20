# pylint: disable=import-error, no-name-in-module

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import logging
import coloredlogs
from dataclasses import asdict

from qiskit.circuit.library import TwoLocal
from qiskit.opflow import PauliSumOp, AerPauliExpectation
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

from src.expectation import Expectation
from src.fidelity import Fidelity

from src.gradients import ExpectationReverse, FidelityReverse
from src.optimizers import GradientDescent
from src.realdual import DualRTE

coloredlogs.install(level="INFO", logger=logging.getLogger("src.realdual"))


reps = 1
# dirname = "data/heisen2/lrel1e1/sv/--_gd100.npy"
dirname = "ising_long.npy"
filenames = [dirname]
# filenames = [f"{dirname}/{i}.npy" for i in range(1, reps + 1)]
# filenames = ["data/heisen3/lr1e3/spsa_it100_b1/sv_xi1.npy"]

J = 1 / 4
h = 1
hamiltonian = PauliSumOp.from_list([("ZZI", J), ("IZZ", J), ("XII", h), ("IXI", h), ("IIX", h)])
magn = PauliSumOp.from_list([("ZII", 1 / 3), ("IZI", 1 / 3), ("IIZ", 1 / 3)])

circuit = TwoLocal(3, rotation_blocks="rx", entanglement_blocks="rzz", reps=2)
initial_parameters = np.zeros(circuit.num_parameters)

print(circuit.draw())

final_time = 0.02
timestep = 0.01

backend = AerSimulator(method="statevector")
expectation_converter = AerPauliExpectation()
shots = None
expectation_gradient = ExpectationReverse()
fidelity_gradient = FidelityReverse()

expectation = Expectation(backend, expectation_converter, shots=shots)
fidelity = Fidelity(backend, expectation_converter, shots=shots)


opt = GradientDescent(maxiter=100, learning_rate=0.1, blocking=False, losstol=0, gradtol=0)
# opt = SPSA(maxiter=100, learning_rate=0.1, perturbation=0.1, batch_size=1, averaging=1)

# algo = VarQITE(circuit, initial_parameters, backend)
algo = DualRTE(
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
    result = algo.evolve(hamiltonian, final_time, timestep, magn)
    print(result.energies[-1])
    np.save(filename, asdict(result))
