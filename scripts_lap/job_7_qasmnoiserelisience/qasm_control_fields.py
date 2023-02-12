import sys

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
from qiskit.providers.fake_provider import FakeMelbourne
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import Estimator, QiskitRuntimeService, Session
from qiskit_ibm_runtime.options import Options

save_path = ""

num_qubits = 4
learning_locality = 3

horiginal = lattice_hamiltonian(
    num_qubits, 1 / 4, -1, one_local=["Z"], two_local=["XX", "YY", "ZZ"]
)

if len(sys.argv) > 1:
    coeffs = sys.argv[1::2]
    terms = sys.argv[2::2]
    control_field = SparsePauliOp.from_list(
        [(terms[i], float(coeffs[i])) for i in range(len(coeffs))]
    )
    horiginal = (horiginal + control_field).simplify()

coriginal = KLocalPauliBasis(learning_locality, num_qubits).pauli_to_vector(horiginal)

ansatz_arguments = {
    "num_qubits": num_qubits,
    "depth": 2,
    "entanglement": "reverse_linear",
    "su2_gates": ["ry"],
    "ent_gates": ["cx"],
}
ansatz, x0 = efficientTwoLocalansatz(**ansatz_arguments)
varqite_kwargs = {"ode_solver": ForwardEulerSolver, "num_timesteps": 3}
beta_timestep = 0.1
beta = varqite_kwargs["num_timesteps"] * beta_timestep * 2

problem = TimeEvolutionProblem(hamiltonian=horiginal ^ "I" * num_qubits, time=beta / 2)

#########################Problem defined. Now set up the backend.
# load the service and set the backend to the simulator
service = QiskitRuntimeService()
backend = "ibmq_qasm_simulator"
# Make a noise model
fake_backend = FakeMelbourne()
noise_model = NoiseModel.from_backend(fake_backend)

# Set options to include the noise model
options = Options()
options.simulator = {
    "noise_model": noise_model,
    "basis_gates": fake_backend.configuration().basis_gates,
    "coupling_map": fake_backend.configuration().coupling_map,
    "seed_simulator": 42,
}

# Set number of shots, optimization_level and resilience_level
options.execution.shots = 1e4
options.optimization_level = 2
options.resilience_level = 2

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

print(result_varqite.times)

#########################Storing result
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
gibbs_result.save(save_path + f"num_qubits{num_qubits}_controlfield={sys.argv[1:]}")
