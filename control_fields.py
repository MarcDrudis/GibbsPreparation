import numpy as np
from dualtime.src.mclachlan import VarQITE
from gibbs.utils import create_hamiltonian_lattice, create_heisenberg
from gibbs.preparation.varqite import efficientTwoLocalansatz
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.dataclass import GibbsResult
import sys

save_path = "saved_simulations/turbo/heisenberg_MAP/"

num_qubits = 8



horiginal = create_heisenberg(num_qubits,1/4,-1)

if len(sys.argv) > 1:
    coeffs = sys.argv[1::2]
    terms = sys.argv[2::2]
    control_field = SparsePauliOp.from_list([(terms[i],float(coeffs[i])) for i in range(len(coeffs))])
    horiginal = (horiginal+control_field).simplify()

coriginal = KLocalPauliBasis(2,num_qubits).pauli_to_vector(horiginal)

ansatz_arguments = {"num_qubits":num_qubits,"depth":2,"entanglement":"reverse_linear","su2_gates":["ry"],"ent_gates":["cx"]}
ansatz,x0 = efficientTwoLocalansatz(**ansatz_arguments)
varqite = VarQITE(ansatz,x0,backend="TURBO")

beta= 1
steps = 20
result_varqite = varqite.evolve(PauliSumOp(horiginal^("I"*num_qubits)),beta/2,timestep = beta/(2*steps))

gibbs_result = GibbsResult(ansatz_arguments=ansatz_arguments,
                        parameters=result_varqite.parameters,
                        coriginal=coriginal,
                        num_qubits=num_qubits,
                        klocality=2,
                        betas = [2 *t for t in result_varqite.times] 
)

gibbs_result.save(save_path+"cfield="+str(sys.argv[1:]))