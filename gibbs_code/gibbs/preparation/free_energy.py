from __future__ import annotations

from cmaes import CMA
from qiskit.quantum_info import Statevector, entropy, partial_trace
import numpy as np


def brute_cmaes(ansatz, hamiltonian, beta, max_evals):
    """Returns the parameters of the ansatz at a thermal state with temperature `beta`.
    It uses CMAES to find the parameters that minimize the free energy.
    """
    loss = lambda parameters: free_energy(ansatz, parameters, hamiltonian, beta)
    N = hamiltonian.num_qubits
    initial_point = np.random.rand(ansatz.num_parameters)
    optimizer = CMA(
        mean=initial_point,
        sigma=1,
        bounds=np.array([[-np.pi, np.pi]] * ansatz.num_parameters),
    )

    fevals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            fevals += 1
            x = optimizer.ask()
            value = loss(x)
            solutions.append((x, value))
        optimizer.tell(solutions)
        if optimizer.should_stop() or fevals >= max_evals:
            break
    # return optimizer._mean,loss(optimizer._mean), fevals
    return x, loss(x), fevals


def free_energy(ansatz, parameters, hamiltonian, beta):
    N = hamiltonian.num_qubits
    state = partial_trace(
        Statevector(ansatz.bind_parameters(parameters)), range(N, 2 * N)
    )
    energy = state.expectation_value(hamiltonian)
    entr = entropy(state, base=np.e)
    return np.real(energy - entr / beta)
