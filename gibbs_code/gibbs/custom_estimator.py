import signal
import time
from typing import Any, Sequence, Union

from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.providers import JobStatus
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_ibm_runtime import Estimator, QiskitRuntimeService, Session


def timeout_handler(signum, frame):
    raise Exception("Iteration timed out")


class RetryEstimator(Estimator):
    def __init__(
        self, *args, backend: str, max_retries: int = 10, timeout: int = 3600, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.timeout = timeout
        self.backend = backend
        signal.signal(signal.SIGALRM, timeout_handler)

    def run(
        self,
        circuits: Union[QuantumCircuit, Sequence[QuantumCircuit]],
        observables: Union[
            BaseOperator, PauliSumOp, Sequence[Union[BaseOperator, PauliSumOp]]
        ],
        parameter_values: Union[
            Sequence[float], Sequence[Sequence[float]], None
        ] = None,
        **kwargs: Any,
    ):
        result = None
        for i in range(self.max_retries):
            try:
                job = super().run(
                    circuits,
                    observables,
                    parameter_values,
                    **kwargs,
                )
                while job.status() in [
                    JobStatus.INITIALIZING,
                    JobStatus.QUEUED,
                    JobStatus.VALIDATING,
                ]:
                    time.sleep(
                        5
                    )  # Check every 5 seconds whether job status has changed
                signal.alarm(
                    self.timeout
                )  # Once job starts running, set timeout to 1 hour by default
                result = job.result()
                if result is not None:
                    signal.alarm(0)  # reset timer
                    return job
            except Exception as exc:
                print("\nSomething went wrong...")
                print(f"\n\nERROR MESSAGE:\n{exc}\n\n")
                if "job" in locals():  # Sometimes job fails to create
                    print(f"Job ID: {job.job_id}. Job status: {job.status()}.")
                    if job.status() not in [
                        JobStatus.DONE,
                        JobStatus.ERROR,
                        JobStatus.CANCELLED,
                    ]:
                        job.cancel()
                else:
                    print("Failed to create job.")
                print(f"Starting trial number {i+2}...\n")
                print("Creating new session...\n")
                signal.alarm(0)  # reset timer
                self._session = Session(backend=self.backend)
        if result is None:
            raise RuntimeError(
                f"Program failed! Maximum number of retries ({self.max_retries}) exceeded"
            )
