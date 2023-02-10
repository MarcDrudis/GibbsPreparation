import numpy as np
from gibbs.utils import create_hamiltonian_lattice, create_heisenberg
from gibbs.preparation.varqite import efficientTwoLocalansatz
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Pauli
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.dataclass import GibbsResult
import sys
from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit.algorithms.gradients import ReverseEstimatorGradient, ReverseQGT, LinCombQGT, LinCombEstimatorGradient,ParamShiftEstimatorGradient
from qiskit.algorithms.time_evolvers import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers.variational import VarQITE
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session
from qiskit_ibm_runtime.options import Options
from qiskit.primitives import Estimator as OuterEstimator
# from monkey import RuntimeRetryEstimator
from qiskit.algorithms.time_evolvers.variational import ForwardEulerSolver
from qiskit_aer.primitives import Estimator as AerEstimator
from gibbs.custom_estimator import CounterEstimator
# from gibbs.customRK import customRK

save_path = "saved_simulations/turbo/qiskit_testing/"

num_qubits = 3

horiginal = create_heisenberg(num_qubits,1/4,-1)

arguments = dict()

if len(sys.argv) > 2:
    arguments["coeffs"] = sys.argv[2::2]
    arguments["terms"] = sys.argv[3::2]
    control_field = SparsePauliOp.from_list([(arguments["terms"][i],float(arguments["coeffs"][i])) for i in range(len(arguments["coeffs"]))])
    horiginal = (horiginal+control_field).simplify()

if len(sys.argv) > 1:
    arguments["shots"] = int(sys.argv[1])
else:
    arguments["shots"] = None

coriginal = KLocalPauliBasis(2,num_qubits).pauli_to_vector(horiginal)

ansatz_arguments = {"num_qubits":num_qubits,"depth":2,"entanglement":"reverse_linear","su2_gates":["ry"],"ent_gates":["cx"]}
ansatz,x0 = efficientTwoLocalansatz(**ansatz_arguments)
beta= 1
problem = TimeEvolutionProblem(hamiltonian = PauliSumOp(horiginal^("I"*num_qubits)), time = beta/2)


service = QiskitRuntimeService()    
options = Options()
options.execution = {"shots": arguments["shots"]}


#Consider that we need 2*num_qubits + 1 qubits in order to have the ansatz + the mmt qubit 
backend = "ibmq_qasm_simulator"
with Session(service=service, backend=backend) as session:
    estimator = Estimator(session=session, options=options)
    gradient = ParamShiftEstimatorGradient(estimator)
    qgt = LinCombQGT(estimator)
    variational_principle = ImaginaryMcLachlanPrinciple(qgt = qgt,gradient = gradient)

    varqite_kwargs = {
    "ode_solver" : ForwardEulerSolver,
    "num_timesteps" : 5,
    }

    varqite = VarQITE(ansatz,x0, variational_principle=variational_principle, **varqite_kwargs)
    result_varqite = varqite.evolve(problem)



gibbs_result = GibbsResult(ansatz_arguments=ansatz_arguments,
                        parameters=result_varqite.parameter_values,
                        coriginal=coriginal,
                        num_qubits=num_qubits,
                        klocality=2,
                        betas = [2 *t for t in result_varqite.times],
                        shots = arguments["shots"],

)
gibbs_result.save(save_path+"arguments="+str(arguments))