import numpy as np
from gibbs.utils import create_hamiltonian_lattice, create_heisenberg
from gibbs.preparation.varqite import efficientTwoLocalansatz
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.dataclass import GibbsResult
from qiskit.algorithms.time_evolvers.variational import VarQITE
from qiskit.algorithms.time_evolvers import TimeEvolutionProblem

from qiskit.algorithms.time_evolvers.variational import ForwardEulerSolver


num_qubits = 4



horiginal = create_heisenberg(num_qubits,1/4,-1)


coriginal = KLocalPauliBasis(2,num_qubits).pauli_to_vector(horiginal)

ansatz_arguments = {"num_qubits":num_qubits,"depth":2,"entanglement":"reverse_linear","su2_gates":["ry"],"ent_gates":["cx"]}
ansatz,x0 = efficientTwoLocalansatz(**ansatz_arguments)
beta= 1
steps = 20
problem = TimeEvolutionProblem(hamiltonian = horiginal^"I"*num_qubits, time = beta/2)
varqite = VarQITE(ansatz,x0,estimator=None, num_timesteps= 5)



result_varqite = varqite.evolve(problem)
print(result_varqite.parameter_values.shape)

gibbs_result = GibbsResult(ansatz_arguments=ansatz_arguments,
                        parameters=result_varqite.parameter_values,
                        coriginal=coriginal,
                        num_qubits=num_qubits,
                        klocality=2,
                        betas = [2 *t for t in result_varqite.times] 
)