"""Exact imaginary time evolution based on matrix exponentiation."""

from typing import Union
import numpy as np
import scipy as sc

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.opflow import OperatorBase, PauliSumOp

from models.heisenline import get_model

mode = "heisen2"

if mode == "triangle":
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

    initial_parameters = np.zeros(circuit.num_parameters)

elif mode == "comb":
    magn = 1
    interact = 1
    num_qubits = 12
    # coeffs = [np.sign(np.sin(0.01 + np.pi * (i / num_qubits) ** 3)) for i in range(1, num_qubits + 1)]
    # coeffs = [(-1) ** i for i in range(num_qubits)]
    # coeffs = [1 for i in range(num_qubits)]
    # coeffs = [np.sin(0.5 + np.pi / 4 * i) for i in range(num_qubits)]
    coeffs = [np.sin(np.pi / 4 * i) for i in range(num_qubits)]
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

    ev = np.linalg.eigvalsh(hamiltonian.to_matrix())

    entanglement = [[i, i + 1] for i in range(0, num_qubits - 1, 2)]
    entanglement += [[i, i + 1] for i in range(1, num_qubits - 1, 2)]
    circuit = EfficientSU2(num_qubits, reps=1, entanglement=entanglement)
    circuit.entanglement_blocks = "cx"

    initial_parameters = np.zeros(circuit.num_parameters)
    for i in range(num_qubits):
        initial_parameters[~(i + num_qubits)] = np.pi / 2
elif mode == "heisen2":
    hamiltonian, circuit, _ = get_model(num_sites=2, J=0.25, g=-1, reps=1)
    initial_parameters = np.zeros(circuit.num_parameters)
    circuit.x(0)
    circuit.x(1)
    circuit.h([0, 1])
elif mode == "heisen3":
    hamiltonian, circuit, initial_parameters = get_model(num_sites=3, J=0.25, g=-1, reps=3)
elif mode == "heisen8":
    hamiltonian, circuit, initial_parameters = get_model(num_sites=8, J=0.25, g=-1, reps=3)
elif mode == "heisen12":
    hamiltonian, circuit, initial_parameters = get_model(
        num_sites=12, J=0.25, g=-1, reps=3, periodic=True
    )

initial_state = circuit.bind_parameters(initial_parameters)


def evolve(
    initial_state: Union[QuantumCircuit, np.ndarray],
    hamiltonian: OperatorBase,
    final_time: float,
    timestep: float,
):
    hamiltonian_matrix = hamiltonian.to_matrix()
    initial_state = Statevector(initial_state)
    initial_statevector = initial_state.data

    times = [0]
    energies = [initial_state.expectation_value(hamiltonian_matrix)]
    states = [initial_statevector]

    while times[-1] < final_time:
        print(f"Time {times[-1]}/{final_time}")
        times.append(times[-1] + timestep)
        statevector = sc.linalg.expm(-times[-1] * hamiltonian_matrix).dot(initial_statevector)
        statevector /= np.linalg.norm(statevector)
        energies.append(Statevector(statevector).expectation_value(hamiltonian_matrix))
        states.append(statevector)

    return times, energies, states


t_, e_, s_ = evolve(initial_state, hamiltonian, final_time=1, timestep=0.01)
result = {"times": t_, "energies": e_, "states": s_}
np.save("data/heisen2/--_exact.npy", result)
