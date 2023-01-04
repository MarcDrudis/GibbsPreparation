from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace, entropy
from qiskit.circuit.library import EfficientSU2

from scipy.optimize import minimize


from gibbs.utils import *


def efficient_su2_ansatz(
    hamiltonian: SparsePauliOp,
    depth: int,
    entanglement: str = "circular",
    su2_gates=None,
):
    """Returns an ansatz to simuate the imagiary time evolution of a Hamiltonian using the EfficientSU2 ansatz."""
    num_qubits = hamiltonian.num_qubits
    eff = EfficientSU2(
        2 * num_qubits,
        reps=depth,
        entanglement=entanglement,
        insert_barriers=True,
        su2_gates=su2_gates,
    ).decompose()
    qr = QuantumRegister(num_qubits, name="q")
    ancilla = QuantumRegister(num_qubits, name="a")
    ansatz = QuantumCircuit(qr, ancilla)
    ansatz.append(eff, qargs=list(qr) + list(ancilla))
    ansatz.cx(qr, ancilla)
    x0 = np.zeros(ansatz.num_parameters)
    x0[-4 * num_qubits : -3 * num_qubits] = np.ones(num_qubits) * np.pi / 2
    return ansatz, x0


def brute_force_optimization(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    x0: np.array,
    beta: float = 1.0,):
    """Returns the parameters such that the ansatz represents the thermal state purification of the Hamiltonian."""
    
    def free_energy(x):
        """Returns the free energy of the ansatz."""
        mixed_state = state_from_ansatz(ansatz, x)
        free_energy_value = mixed_state.expectation_value(hamiltonian) - entropy(mixed_state,base=np.e)/beta
        print(free_energy_value)
        return free_energy_value
    
    return minimize(free_energy,x0,method = 'COBYLA',tol = 1e-15)