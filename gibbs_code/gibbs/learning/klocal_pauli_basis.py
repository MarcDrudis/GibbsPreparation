from __future__ import annotations

from itertools import product, combinations, chain
import numpy as np

from qiskit.quantum_info import SparsePauliOp


class KLocalPauliBasis:
    def __init__(self, k, num_qubits, periodic: bool = False):
        self.k = min(k, num_qubits)
        self.num_qubits = num_qubits
        self.periodic = periodic
        self._paulis_list = self._compute_paulis_list(k)
        self._pauli_to_num_dict = dict(
            zip(self._paulis_list, range(len(self._paulis_list)))
        )
        self.size = len(self._paulis_list)

    def pauli_to_num(self, pauli: str) -> int:
        if pauli not in self._paulis_list:
            return None
        return self._pauli_to_num_dict[pauli]

    def num_to_pauli(self, num: int) -> str:
        if num >= len(self._paulis_list):
            return None
        return self._paulis_list[num]

    def pauli_to_vector(self, operator: SparsePauliOp) -> np.array:
        """Converts a pauli operator to a vector of coefficients in the pauli basis"""
        vec = np.zeros(len(self._paulis_list), dtype=np.complex128)
        for pauli, coeff in operator.to_list():
            vec[self.pauli_to_num(pauli)] = coeff
        return vec

    def vector_to_pauli_op(self, vector: np.array) -> SparsePauliOp:
        """Converts a vector of coefficients in the pauli basis to a pauli operator"""
        return SparsePauliOp.from_list(
            [(self._paulis_list[i], v) for i, v in enumerate(vector)]
        ).simplify()

    def _compute_paulis_list(self, k: int):
        """Retruns a list of all the klocal pauli operators with self.num_qubits qubits"""
        if k == 1:
            return list(
                chain.from_iterable([self._extend_pauli(pauli) for pauli in "XYZ"])
            )
        edges = product("XYZ", repeat=2)
        if k == 2:
            return self._compute_paulis_list(1) + list(
                chain.from_iterable(
                    [self._extend_pauli("".join(edge)) for edge in edges]
                )
            )

        plist = product("IXYZ", repeat=k - 2)
        plist = [e[0] + "".join(p) + e[1] for e, p in product(edges, plist)]

        return self._compute_paulis_list(k - 1) + list(
            chain.from_iterable([self._extend_pauli(p) for p in plist])
        )

    def _extend_pauli(self, pauli: str):
        nidentities = self.num_qubits - len(pauli)
        periodic_extension = (
            [pauli[-i:] + "I" * nidentities + pauli[:-i] for i in range(1, len(pauli))]
            if self.periodic
            else []
        )
        extension = [
            "I" * i + pauli + "I" * (nidentities - i) for i in range(nidentities + 1)
        ]
        return extension + periodic_extension
