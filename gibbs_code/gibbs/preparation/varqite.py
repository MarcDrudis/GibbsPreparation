from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector,Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace, entropy, Pauli
from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate
from qiskit.circuit import ParameterVector,ParameterExpression
from qiskit.synthesis.evolution.product_formula import evolve_pauli
import random

from scipy.optimize import minimize


from gibbs.utils import *
from gibbs.preparation.pauli_rotation import RPGate


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


def efficient_su2_ansatz_V3(
    hamiltonian: SparsePauliOp,
    depth: int,
    su2_gates=None,
):
    """Returns an ansatz to simuate the imagiary time evolution of a Hamiltonian using the EfficientSU2 ansatz.
    This time we don't add a final layer to encode the identity purification.
    """
    num_qubits = hamiltonian.num_qubits
    entanglement = [[num_qubits+i,i+1] for i in range(num_qubits-1)] + [[i,num_qubits+i] for i in range(num_qubits)] 
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
    x0 = np.zeros(ansatz.num_parameters)
    x0[-8 * num_qubits : -7 * num_qubits] = np.ones(num_qubits) * np.pi / 2
    return ansatz, x0

def pauli_rotations_ansatz(hamiltonian:SparsePauliOp, depth: int,
    entanglement: str = "circular",
    su2_gates=None,
    rotations_number:int=0):
    """Creates an ansatz that implements a series of Pauli rotations.
    Args:
        rotations: A list of Pauli strings to perform rotations on.
    """
    num_qubits = hamiltonian.num_qubits
    qr = QuantumRegister(num_qubits, name="q")
    ancilla = QuantumRegister(num_qubits, name="a")
    ansatz = QuantumCircuit(qr, ancilla)
    
    #This part of the ansatz is the one that contains most of the expressivity
    eff = EfficientSU2(
    2 * num_qubits,
    reps=depth,
    entanglement=entanglement,
    insert_barriers=True,
    su2_gates=su2_gates,
    ).decompose()
    ansatz.append(eff, qargs=list(qr) + list(ancilla))
    ansatz.barrier()
    #This one is the one that prepares the purification of the identity
    ansatz.cx(qr,ancilla)
    ansatz.barrier()
    #This last part is the one that will break symmetries for the identity.
    # rotations = [conjugate_pauli(str(p)) for p in hamiltonian.paulis]
    rotations = random.choices(population=hamiltonian.paulis, weights=np.abs(hamiltonian.coeffs), k=rotations_number)
    rotations = [conjugate_pauli(str(p)) for p in rotations]
    parameters = ParameterVector("thetas",len(rotations))
    for pauli, parameter in zip(rotations,parameters):    
        ansatz.append(RPGate(pauli,parameter),list(qr)+list(ancilla))
        ansatz.barrier()
        
    #Set the initial parameters to be the identity    
    x0 = np.zeros(ansatz.num_parameters)
    x0[-4 * num_qubits : -3 * num_qubits] = np.ones(num_qubits) * np.pi / 2
    
    return ansatz, x0


def brute_force_optimization(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    x0: np.array,
    beta: float = 1.0,
    tol = 1e-15):
    """Returns the parameters such that the ansatz represents the thermal state purification of the Hamiltonian."""
    
    def free_energy(x):
        """Returns the free energy of the ansatz."""
        mixed_state = state_from_ansatz(ansatz, x)
        free_energy_value = mixed_state.expectation_value(hamiltonian) - entropy(mixed_state,base=np.e)/beta
        print(free_energy_value)
        return free_energy_value
    
    return minimize(free_energy,x0,method = 'COBYLA',tol = 1e-15)