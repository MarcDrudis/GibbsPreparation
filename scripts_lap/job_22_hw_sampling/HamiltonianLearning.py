from gibbs.dataclass import get_results
from gibbs.learning.constraint_matrix import ConstraintMatrixFactory
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.options import Options
import numpy as np
from gibbs.custom_estimator import RetryEstimator
from qiskit.providers.fake_provider import FakeAuckland
from qiskit_aer.noise import NoiseModel

import json


#########Load the arguments from a text file.
with open("arguments.txt") as f:
    data = f.read()
input_args = json.loads(data)

result = get_results(input_args["state_path"])[0]
shots = input_args["shots"]
backend_input = input_args["backend_input"]

##########Choose between qasm simulation or real hw

list_fake_backends = ["FakeAuckland"]
list_real_backends = ["ibm_auckland", "ibmq_mumbai"]

if backend_input in list_fake_backends:
    # Choose service and backend
    service = QiskitRuntimeService()
    backend = "ibmq_qasm_simulator"
    # For qasm simulator we need to setup a noise model
    fake_backend = FakeAuckland()
    noise_model = NoiseModel.from_backend(fake_backend)
    options = Options()
    options.simulator = {
        "noise_model": noise_model,
        "basis_gates": fake_backend.operation_names,
        "coupling_map": sorted(
            set([tuple(sorted(x)) for x in fake_backend.coupling_map])
        ),
        "seed_simulator": 42,
    }

elif backend_input in list_real_backends:
    # Choose service and backend
    backend = backend_input
    service = QiskitRuntimeService()
    options = Options()

else:
    raise ValueError(
        "The backend provided is not in the list of real or fake backends."
    )
##########Set shots and resilience. 2 is ZNE
options.optimization_level = 2
options.resilience_level = 1
###Run the estimator##########################################

cmat = ConstraintMatrixFactory(
    num_qubits=result.num_qubits, k_learning=2, k_constraints=3
)
observables_list = [p + "I" * len(p) for p in cmat.sampling_basis.paulis_list]
ansatz_list = [result.ansatz] * len(observables_list)
parameters_list = [result.parameters[-1]] * len(observables_list)


with Session(service=service, backend=backend):
    estimator = RetryEstimator(
        backend=backend, service=service, timeout=2 * 3600, options=options
    )
    estimator_result = estimator.run(
        circuits=ansatz_list,
        observables=observables_list,
        parameter_values=parameters_list,
        shots=shots,
    ).result()

values = estimator_result.values
variances = [m["variance"] for m in estimator_result.metadata]

np.save(
    "sampled_paulis.npy", {"values": values, "variances": variances}, allow_pickle=True
)
