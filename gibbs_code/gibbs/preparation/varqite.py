from __future__ import annotations

import random

import numpy as np
from gibbs.preparation.pauli_rotation import RPGate
from gibbs.utils import state_from_ansatz
from qiskit.circuit import (
    ClassicalRegister,
    Parameter,
    ParameterExpression,
    ParameterVector,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate, TwoLocal
from qiskit.quantum_info import SparsePauliOp, Statevector, entropy, partial_trace
from qiskit.synthesis.evolution.product_formula import evolve_pauli
from scipy.optimize import minimize


def brute_force_optimization(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    x0: np.array,
    beta: float = 1.0,
    tol=1e-15,
):
    """Returns the parameters such that the ansatz represents the thermal state purification of the Hamiltonian."""

    def free_energy(x):
        """Returns the free energy of the ansatz."""
        mixed_state = state_from_ansatz(ansatz, x)
        free_energy_value = (
            mixed_state.expectation_value(hamiltonian)
            - entropy(mixed_state, base=np.e) / beta
        )
        print(free_energy_value)
        return free_energy_value

    return minimize(free_energy, x0, method="COBYLA", tol=1e-15)


def efficientTwoLocalansatz(
    num_qubits: int,
    depth: int,
    entanglement: str = "circular",
    su2_gates: list[str] = ["rz", "ry"],
    ent_gates: list[str] = ["cx"],
    barriers: bool = False,
    no_hadamart: bool = False,
):
    """Creates an ansatz that implements a series of Pauli rotations.
    Args:
        rotations: A list of Pauli strings to perform rotations on.
    """
    qr = QuantumRegister(num_qubits, name="q")
    ancilla = QuantumRegister(num_qubits, name="a")
    ansatz = QuantumCircuit(qr, ancilla)

    eff = TwoLocal(
        2 * num_qubits,
        su2_gates,
        ent_gates,
        entanglement=entanglement,
        reps=depth,
        insert_barriers=barriers,
        skip_final_rotation_layer=False,
    ).decompose()

    reordered = [None] * num_qubits * 2
    reordered[::2] = list(qr)
    reordered[1::2] = list(ancilla)
    ansatz.append(eff, qargs=reordered)
    # This one is the one that prepares the purification of the identity
    if barriers:
        ansatz.barrier()
    x0 = np.zeros(ansatz.num_parameters)
    if no_hadamart:
        x0[-2 * num_qubits :: 2] = np.pi / 2
    else:
        ansatz.h(qr)
    ansatz.cx(qr, ancilla)

    # Set the initial parameters to be the identity
    if not np.isclose(
        partial_trace(
            Statevector(ansatz.bind_parameters(x0)), range(1, 2 * num_qubits, 2)
        ).data,
        np.eye(2**num_qubits) / 2**num_qubits,
    ).all():
        raise ValueError("This configuration of ansatz does not start in identity.")
    return ansatz, x0
