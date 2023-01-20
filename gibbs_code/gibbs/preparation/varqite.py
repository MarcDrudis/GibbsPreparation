from __future__ import annotations

from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector,Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace, entropy, Pauli
from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate, TwoLocal
from qiskit.circuit import ParameterVector,ParameterExpression
from qiskit.synthesis.evolution.product_formula import evolve_pauli
import random

from scipy.optimize import minimize


from gibbs.utils import *
from gibbs.preparation.pauli_rotation import RPGate

# def pauli_rotations_ansatz(hamiltonian:SparsePauliOp, depth: int,
#     entanglement: str = "circular",
#     su2_gates:list[str]=["rz","ry"],
#     ent_gates:list[str]=["cx"],
#     rotations: list[str]=[]):
#     """Creates an ansatz that implements a series of Pauli rotations.
#     Args:
#         rotations: A list of Pauli strings to perform rotations on.
#     """
#     num_qubits = hamiltonian.num_qubits
#     qr = QuantumRegister(num_qubits, name="q")
#     ancilla = QuantumRegister(num_qubits, name="a")
#     # mixed_registers = [None]*2*num_qubits
#     # mixed_registers[::2] = list(qr)
#     # mixed_registers[1::2] = list(ancilla)
#     ansatz = QuantumCircuit(qr,ancilla)
    
#     #This part of the ansatz is the one that contains most of the expressivity
#     # eff = EfficientSU2(
#     # 2 * num_qubits,
#     # reps=depth,
#     # entanglement=entanglement,
#     # insert_barriers=True,
#     # su2_gates=su2_gates,
#     # ).decompose()
    
#     eff = TwoLocal(2*num_qubits,su2_gates,ent_gates,entanglement=entanglement, reps=depth, insert_barriers=True).decompose()

#     ansatz.append(eff, qargs=list(qr) + list(ancilla))
#     ansatz.barrier()
#     #This one is the one that prepares the purification of the identity
#     ansatz.h(qr)
#     ansatz.cx(qr,ancilla)
#     ansatz.barrier()
#     #This last part is the one that will break symmetries for the identity.
#     # rotations = [conjugate_pauli(str(p)) for p in hamiltonian.paulis]
#     # rotations = random.choices(population=hamiltonian.paulis, weights=np.abs(hamiltonian.coeffs), k=rotations_number)
    
#     rotations = [conjugate_pauli(str(p)) for p in rotations]
#     parameters = ParameterVector("thetas",len(rotations))
#     for pauli, parameter in zip(rotations,parameters):    
#         ansatz.append(RPGate(pauli,parameter),list(qr)+list(ancilla))
#         ansatz.barrier()
        
#     #Set the initial parameters to be the identity    
#     x0 = np.zeros(ansatz.num_parameters)
    
#     return ansatz, x0


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

def efficientTwoLocalansatz(num_qubits:int,
                            depth: int,
                            entanglement: str = "circular",
                            su2_gates:list[str]=["rz","ry"],
                            ent_gates:list[str]=["cx"]):
    """Creates an ansatz that implements a series of Pauli rotations.
    Args:
        rotations: A list of Pauli strings to perform rotations on.
    """
    qr = QuantumRegister(num_qubits, name="q")
    ancilla = QuantumRegister(num_qubits, name="a")
    ansatz = QuantumCircuit(qr,ancilla)
    
    eff = TwoLocal(2*num_qubits,su2_gates,ent_gates,entanglement=entanglement, reps=depth, insert_barriers=True).decompose()

    ansatz.append(eff, qargs=list(qr) + list(ancilla))
    ansatz.barrier()
    #This one is the one that prepares the purification of the identity
    ansatz.h(qr)
    ansatz.cx(qr,ancilla)
    ansatz.barrier()
        
    #Set the initial parameters to be the identity    
    x0 = np.zeros(ansatz.num_parameters)
    
    return ansatz, x0
    