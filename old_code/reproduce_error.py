import numpy as np
from qiskit.algorithms.gradients import LinCombQGT,LinCombEstimatorGradient
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

service = QiskitRuntimeService()    
qc = EfficientSU2(2, reps=1)
x0 = np.zeros(qc.num_parameters)
obs = SparsePauliOp.from_list([("II",1)])

with Session(service=service, backend="ibmq_qasm_simulator") as session:
    estimator = Estimator(session=session)
    # print(estimator.run([qc],[obs],[x0]).result())
    # gradient = LinCombEstimatorGradient(estimator)
    qgt = LinCombQGT(estimator)
    print(qgt.run([qc],[x0]).result())