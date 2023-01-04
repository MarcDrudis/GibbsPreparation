import sys
sys.path.append("~/GH/Julien/gibbs_state_preparation/")
import numpy as np

from qiskit.circuit import QuantumCircuit,QuantumRegister,ParameterVector
from qiskit.quantum_info import SparsePauliOp,Statevector, partial_trace, entropy
from qiskit.circuit.library import EfficientSU2

from scipy.optimize import minimize


from .utils import *
# from pauli_rotation import RPGate

def entropy_conserving_ansatz_V2(
    hamiltonian: SparsePauliOp,
    random_tensor: np.array,
    reps: int = 2,
    derivatve: int = -1,
):
    """
    Creates an entropy conserving ansatz.
    Args:
        hamiltonian: The Hamiltonian to be simulated.
        random_tensor: Contains the Pauli rotations associated to each parameter. [X or Y, n_qubit, n_param]
    """
    N = hamiltonian.num_qubits
    qr = QuantumRegister(N, name="q")
    ancilla = QuantumRegister(N, name="a")
    circuit = QuantumCircuit(qr, ancilla)

    Rotations = ParameterVector("R", random_tensor.shape[2])

    circuit.h(qr)
    circuit.cx(qr, ancilla)
    circuit.barrier()

    if derivatve >= 0:
        for i in range(N):
            if random_tensor[0, i, derivatve] == 1:
                circuit.x(qr[i])
            if random_tensor[0, N + i, derivatve] == 1:
                circuit.x(ancilla[i])
                circuit.barrier()

        for i in range(N):
            if random_tensor[1, i, derivatve] == 1:
                circuit.y(qr[i])
            if random_tensor[1, N + i, derivatve] == 1:
                circuit.y(ancilla[i])

        circuit.barrier()
        for i in range(N):
            if random_tensor[2, i, derivatve] == 1:
                circuit.z(qr[i])
            if random_tensor[2, N + i, derivatve] == 1:
                circuit.z(ancilla[i])

    else:
        for i in range(N):
            xrot_qubit = [
                rot_param * random_tensor[0, i, k]
                for k, rot_param in enumerate(Rotations)
            ]
            circuit.rx(sum(xrot_qubit), qr[i])
            yrot_qubit = [
                rot_param * random_tensor[1, i, k]
                for k, rot_param in enumerate(Rotations)
            ]
            circuit.ry(sum(yrot_qubit), qr[i])
            zrot_qubit = [
                rot_param * random_tensor[2, i, k]
                for k, rot_param in enumerate(Rotations)
            ]
            circuit.rz(sum(zrot_qubit), qr[i])

            xrot_ancilla = [
                rot_param * random_tensor[0, N + i, k]
                for k, rot_param in enumerate(Rotations)
            ]
            circuit.rx(sum(xrot_ancilla), ancilla[i])
            yrot_qubit = [
                rot_param * random_tensor[1, N + i, k]
                for k, rot_param in enumerate(Rotations)
            ]
            circuit.ry(sum(yrot_qubit), ancilla[i])
            zrot_qubit = [
                rot_param * random_tensor[2, N + i, k]
                for k, rot_param in enumerate(Rotations)
            ]
            circuit.rz(sum(zrot_qubit), qr[i])

    circuit.barrier()

    # effsu2 = EfficientSU2(N,reps=reps,insert_barriers=True,entanglement="circular").decompose()
    # circuit.append(effsu2,qr)

    return circuit


def efficient_su2_ansatz(hamiltonian,depth:int,entanglement:str="circular",su2_gates = None):
    num_qubits = hamiltonian.num_qubits
    eff = EfficientSU2(2*num_qubits, reps=depth, entanglement=entanglement, insert_barriers=True, su2_gates=su2_gates).decompose()
    qr = QuantumRegister(num_qubits, name="q")
    ancilla = QuantumRegister(num_qubits, name="a")
    ansatz = QuantumCircuit(qr,ancilla)
    ansatz.append(eff,qargs=list(qr)+list(ancilla))
    ansatz.cx(qr, ancilla)
    x0 = np.zeros(ansatz.num_parameters)
    x0[-4*num_qubits:-3*num_qubits] = np.ones(num_qubits)*np.pi/2
    return ansatz,x0

# def efficient_su2_ansatz_V2(hamiltonian,depth:int,entanglement:str="circular"):
#     ansatz,x0 = efficient_su2_ansatz(hamiltonian,depth,entanglement)
#     paulis = [str(h) for h in hamiltonian.paulis]
#     pauli_rots = ParameterVector("P", len(paulis))
#     for h,p in zip(paulis,pauli_rots):
#         ansatz.append(RPGate(str(h),p),range(hamiltonian.num_qubits))
    
#     return ansatz,x0


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





# if __name__ == "__main__":
    # hamiltonian = SparsePauliOp.from_list([("XYZ", 1.0)])
    # ansatz,x0 = efficient_su2_ansatz_V2(hamiltonian,depth=2,entanglement="circular")
    # ansatz.draw(output="text")

