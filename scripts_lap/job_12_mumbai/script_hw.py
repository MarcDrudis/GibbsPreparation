import json

from gibbs.dataclass import GibbsResult
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.preparation.varqite import efficientTwoLocalansatz
from gibbs.qfiwrapper import variationalprinciplestorage
from gibbs.utils import lattice_hamiltonian
from qiskit.algorithms.gradients import LinCombEstimatorGradient, LinCombQGT
from qiskit.algorithms.time_evolvers import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers.variational import (
    ForwardEulerSolver,
    ImaginaryMcLachlanPrinciple,
    VarQITE,
)
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Estimator, QiskitRuntimeService, Session
from qiskit_ibm_runtime.options import Options

#########Load the arguments from a text file.
with open("arguments.txt") as f:
    data = f.read()
input_args = json.loads(data)
#########Set the variables inside of the python script.
num_qubits = input_args["num_qubits"]
learning_locality = input_args["learning_locality"]
beta_timestep = input_args["beta_timestep"]
num_timesteps = input_args["num_timesteps"]
ansatz_arguments = {
    k: input_args[k]
    for k in (
        "num_qubits",
        "depth",
        "entanglement",
        "su2_gates",
        "ent_gates",
    )
}
lattice_hamiltonian_arguments = {
    k: input_args[k]
    for k in (
        "num_qubits",
        "j_const",
        "g_const",
        "one_local",
        "two_local",
    )
}
varqite_kwargs = {"ode_solver": ForwardEulerSolver, "num_timesteps": num_timesteps}
beta = varqite_kwargs["num_timesteps"] * beta_timestep

backend = input_args["backend"]

##########Initialize the problem.
horiginal = lattice_hamiltonian(**lattice_hamiltonian_arguments)
if input_args["cfield"] is not None:
    control_field = SparsePauliOp.from_list(
        list(zip(input_args["cfield"][::2], input_args["cfield"][1::2]))
    )
    horiginal = (horiginal + control_field).simplify()
coriginal = KLocalPauliBasis(learning_locality, num_qubits).pauli_to_vector(horiginal)
ansatz, x0 = efficientTwoLocalansatz(**ansatz_arguments)
problem = TimeEvolutionProblem(hamiltonian=horiginal ^ "I" * num_qubits, time=beta / 2)
##########Choose service and backend
service = QiskitRuntimeService()
options = Options()
##########Set shots and resilience. 2 is ZNE
options.execution.shots = 2e3
options.optimization_level = 2
options.resilience_level = 2
options.resilience.noise_amplifier = "LocalFoldingAmplifier"
options.resilience.noise_factors = (1, 3, 5)
##########Initialize Estimator and Gradients and run the simulation
with Session(service=service, backend=backend):
    estimator = Estimator(options=options)
    gradient = LinCombEstimatorGradient(estimator)
    qgt = LinCombQGT(estimator)
    variational_principle = variationalprinciplestorage(ImaginaryMcLachlanPrinciple)(
        gradient=gradient, qgt=qgt
    )

    varqite = VarQITE(
        ansatz, x0, variational_principle=variational_principle, **varqite_kwargs
    )
    print("Evolving")
    result_varqite = varqite.evolve(problem)

print(2 * result_varqite.times)


########## Storing result
gibbs_result = GibbsResult(
    ansatz_arguments=ansatz_arguments,
    parameters=result_varqite.parameter_values,
    coriginal=coriginal,
    num_qubits=num_qubits,
    klocality=learning_locality,
    betas=[2 * t for t in result_varqite.times],
    stored_qgts=variational_principle.stored_qgts,
    stored_gradients=variational_principle.stored_gradients,
)
gibbs_result.save("hardware_result")
