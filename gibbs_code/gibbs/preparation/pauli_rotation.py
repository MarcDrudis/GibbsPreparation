"""Rotation around the arbitrary Pauli axis."""

from __future__ import annotations

from typing import Optional

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.synthesis.evolution.product_formula import (
    diagonalizing_clifford,
    cnot_chain,
    cnot_fountain,
)
from scipy.linalg import expm
from qiskit.quantum_info import Pauli
import numpy as np


class RPGate(Gate):
    r"""Single-qubit rotation about the arbitrary Pauli axis."""

    def __init__(
        self,
        pauli: str,
        theta: ParameterValueType,
        label: Optional[str] = None,
        cx_structure: str = "chain",
    ):
        """Create new RPauli gate."""
        super().__init__(f"RPgate({pauli})", len(pauli), [theta], label=label)
        self.pauli = pauli
        self.cx_structure = cx_structure

    def _define(self):
        """ """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from qiskit.circuit.library.standard_gates.x import CXGate
        from qiskit.circuit.library.standard_gates.rz import RZGate
        from qiskit.circuit.library.standard_gates.ry import RYGate
        from qiskit.circuit.library.standard_gates.rx import RXGate

        # q_0: ──■─────────────■──
        #      ┌─┴─┐┌───────┐┌─┴─┐
        # q_1: ┤ X ├┤ Rz(0) ├┤ X ├
        #      └───┘└───────┘└───┘
        q = QuantumRegister(len(self.pauli), "q")
        theta = self.params[0]
        qc = QuantumCircuit(q, name=self.name)

        # The roation will be performed on the last qubit that is non trivial in the pauli string.
        for i, p in enumerate(self.pauli):
            if p != "I":
                rotation_qubit = len(self.pauli) - i - 1
                break
        rot_rules = [(RZGate(theta), [q[rotation_qubit]], [])]

        # We need to rotate the qubits to the basis in which the rotation is performed.
        basis_rotation = {
            "I": None,
            "X": RYGate(np.pi / 2),
            "Y": RXGate(np.pi / 2),
            "Z": None,
        }
        basis_rules = [
            (basis_rotation[p], [q[i]], [])
            for i, p in enumerate(self.pauli[::-1])
            if basis_rotation[p] is not None
        ]

        # The entanglement will be performed only on non-trivial qubits.

        ent_rules = [
            (CXGate(), [q[i], q[rotation_qubit]], [])
            for i in range(rotation_qubit)
            if self.pauli[::-1][i] != "I"
        ]

        rules = basis_rules + ent_rules + rot_rules + ent_rules[::-1] + basis_rules

        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted RX gate.

        :math:`RX(\lambda)^{\dagger} = RX(-\lambda)`
        """
        return RPGate(self.pauli, -self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the RX gate."""

        # return expm(-1.0j*self.params[0]* Pauli(self.pauli).to_matrix())
        return (
            np.cos(self.params[0] / 2) * np.eye(2 ** len(self.pauli))
            - 1j * np.sin(self.params[0] / 2) * Pauli(self.pauli).to_matrix()
        )
