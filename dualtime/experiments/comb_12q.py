# pylint: disable=import-error, no-name-in-module

import set_paths  # pylint: disable=unused-import

import numpy as np
import matplotlib.pyplot as plt
import logging
import coloredlogs
from dataclasses import asdict

from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from qiskit.opflow import PauliSumOp, AerPauliExpectation, PauliExpectation
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

from src.expectation import Expectation
from src.fidelity import Fidelity

from src.gradients import ExpectationParameterShift, FidelityParameterShift
from src.gradients import ExpectationReverse, FidelityReverse
from src.dual import DualITE
from src.mclachlan import VarQITE

from surfer.gradient import ReverseGradient
from surfer.qfi import ReverseQFI

coloredlogs.install(level="INFO", logger=logging.getLogger("src.dual"))

# reps = 5
# dirname = "data/comb/shots1024_xi1_ws"
# filenames = [f"{dirname}/{i}.npy" for i in range(1, reps + 1)]
filenames = ["data/comb/varqite_real.npy"]

num_qubits = 12
# coeffs = [np.sign(np.sin(0.01 + np.pi * (i / num_qubits) ** 3)) for i in range(1, num_qubits + 1)]
# coeffs = [(-1) ** i for i in range(num_qubits)]
# coeffs = [1 for i in range(num_qubits)]
coeffs = [np.sin(0.5 + np.pi / 4 * i) for i in range(num_qubits)]
seed = 2
np.random.seed(seed)
# coeffs = np.random.random_integers(-10, 10, size=num_qubits - 1)
hamiltonian = PauliSumOp(
    SparsePauliOp.from_sparse_list(
        [("ZZ", [i, i + 1], 1) for i in range(num_qubits - 1)]
        + [("Z", [i], coeffs[i]) for i in range(num_qubits)],
        # + [("ZZ", [num_qubits - 1, 0], coeffs[-1])],
        # [("ZZ", [0, 1], 1), ("Z", [0], 1)],
        num_qubits,
    )
)
print(hamiltonian)
ev = np.linalg.eigvalsh(hamiltonian.to_matrix())
print(list(sorted(ev)))
# exit()

# circuit = RealAmplitudes(3, reps=1, entanglement="reverse_linear")
entanglement = [[i, i + 1] for i in range(0, num_qubits - 1, 2)]
entanglement += [[i, i + 1] for i in range(1, num_qubits - 1, 2)]
# entanglement += [[num_qubits - 1, 0]]
circuit = RealAmplitudes(num_qubits, reps=1, entanglement=entanglement)
# circuit.rotation_blocks = "cry"
# circuit = EfficientSU2(num_qubits, reps=1, entanglement=entanglement)
# circuit.entanglement_blocks = "cz"
# circuit = RealAmplitudes(4, reps=0)
# circuit.rotation_blocks = ["rz", "ry", "rz"]

# rx = TwoLocal(
#     num_qubits,
#     "ry",
#     "crx",
#     reps=1,
#     entanglement=entanglement,
#     skip_final_rotation_layer=True,
#     parameter_prefix="x",
# )
# ry = TwoLocal(
#     num_qubits,
#     ["ry", "rz"],
#     "cz",
#     reps=0,
#     entanglement="pairwise",
#     skip_final_rotation_layer=False,
#     parameter_prefix="y",
# )
# circuit = rx.compose(ry)

initial_parameters = np.zeros(circuit.num_parameters)
for i in range(num_qubits):
    initial_parameters[~i] = (1) ** 1 * np.pi / 2
    # initial_parameters[~(i + num_qubits)] = np.pi / 2

print(circuit.decompose().bind_parameters(initial_parameters).draw())

b0 = -ReverseGradient().compute(hamiltonian, circuit, initial_parameters)
print(b0)
g0 = ReverseQFI().compute(circuit, initial_parameters) / 4

import matplotlib.pyplot as plt

plt.imshow(g0)
plt.show()

x0 = np.linalg.solve(g0 + np.identity(b0.size) * 0.01, b0)
print(x0)

final_time = 2
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


algo = VarQITE(circuit, initial_parameters, backend)
# algo = DualITE(
#     circuit,
#     initial_parameters,
#     expectation,
#     fidelity,
#     expectation_gradient,
#     fidelity_gradient,
#     warmstart=x0,
# )

for filename in filenames:
    result = algo.evolve(hamiltonian, final_time, timestep)
    np.save(filename, asdict(result))

print(result.parameters[-1])
print(circuit.bind_parameters(result.parameters[-1]).decompose().draw())
plt.plot(result.times, result.energies)
plt.show()
