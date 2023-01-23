import unittest
from ddt import data, ddt, unpack
from qiskit.circuit import QuantumCircuit, Parameter
from gibbs.preparation.pauli_rotation import RPGate


@ddt
class TeestPauliRotation(unittest.TestCase):
    def create_RPGate(self, num_qubits, pauli):
        """Creates a quantum circuit and appends an RPGate to it."""
        qr = QuantumRegister(num_qubits, name="q")
        theta = Parameter("theta")
        circuit = QuantumCircuit(qr)
        circuit.append(RPGate(pauli, theta), qargs=qr)
        return circuit

    @data([3, "XYZ"])
    @unpack
    def test_pauli_rotation(self, num_qubits, pauli):
        circuit = self.create_RPGate(num_qubits, pauli)
