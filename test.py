import numpy as np
from dualtime.src.mclachlan import VarQITE
from gibbs_state_preparation.utils import *
from gibbs_state_preparation.varqite_gibbs_preparation import efficient_su2_ansatz
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

beta = 1.0
hamiltonian = create_hamiltonian_lattice(3,1.3,1.7)
hamiltonian = SparsePauliOp.from_list([("XX",1)])
N = hamiltonian.num_qubits
ansatz,x0 = efficient_su2_ansatz(hamiltonian,depth = 2,entanglement='reverse_linear')
varqite = VarQITE(ansatz,x0,backend=None)
result = varqite.evolve(PauliSumOp(hamiltonian^("I"*N)),beta/2,timestep = 0.01)
print(result.times)
print(result.energies)

final_state = state_from_ansatz(ansatz,result.parameters[-1])

print("The max difference is:", np.max(np.abs(final_state-expected_state(hamiltonian,beta))))
printarray(final_state,3,np.abs)
printarray(expected_state(hamiltonian,beta),3,np.abs)
# np.testing.assert_almost_equal(final_state,expected_state(hamiltonian,beta),decimal=5)