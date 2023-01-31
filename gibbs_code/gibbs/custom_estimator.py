from qiskit.primitives import Estimator
from collections.abc import Sequence
from qiskit.primitives.base import EstimatorResult


class CounterEstimator(Estimator):
    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        print(run_options)
        super()._call(
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            **run_options,
        )
