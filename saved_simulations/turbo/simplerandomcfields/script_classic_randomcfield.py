import json
import sys

import numpy as np
from gibbs.dataclass import GibbsResult
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.preparation.varqite import efficientTwoLocalansatz
from gibbs.qfiwrapper import variationalprinciplestorage
from gibbs.utils import lattice_hamiltonian
from qiskit.algorithms.gradients import ReverseEstimatorGradient, ReverseQGT
from qiskit.algorithms.time_evolvers import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers.variational import (
    ForwardEulerSolver,
    ImaginaryMcLachlanPrinciple,
    VarQITE,
)
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import random

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
add_cfield = bool(input_args["add_cfield"])
##########Initialize the problem.
horiginal = lattice_hamiltonian(**lattice_hamiltonian_arguments)
kbasis = KLocalPauliBasis(learning_locality, num_qubits)
coriginal = kbasis.pauli_to_vector(horiginal)
if add_cfield:
    coriginal[random.randint(0, KLocalPauliBasis(1, num_qubits).size)] = random.uniform(
        -0.2, 0.2
    )
    horiginal = kbasis.vector_to_pauli_op(coriginal)
ansatz, x0 = efficientTwoLocalansatz(**ansatz_arguments)
problem = TimeEvolutionProblem(hamiltonian=horiginal ^ "I" * num_qubits, time=beta / 2)

##########Initialize the Estimator and Gradients and perform Evolution
estimator = Estimator()
gradient = ReverseEstimatorGradient()
qgt = ReverseQGT()
variational_principle = variationalprinciplestorage(ImaginaryMcLachlanPrinciple)(
    gradient=gradient, qgt=qgt
)

varqite = VarQITE(
    ansatz, x0, variational_principle=variational_principle, **varqite_kwargs
)
print("Evolving")
result_varqite = varqite.evolve(problem)

print(2 * result_varqite.times)

##########Storing result
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
gibbs_result.save("classic_result")
