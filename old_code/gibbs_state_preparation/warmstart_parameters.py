#%%
from gibbs_state_preparation import *
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace
import numpy as np

from hamiltonian_learning import create_hamiltonian_lattice
from qiskit.primitives import Estimator
from cmaes import CMA

import numpy as np
from qiskit.algorithms.time_evolvers.variational.var_qite import VarQITE
from qiskit.algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.utils import algorithm_globals
from qiskit.primitives import Estimator
from qiskit.algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
)
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector
from scipy.sparse.linalg import expm_multiply, expm
from scipy.linalg import logm

from qiskit.primitives import Sampler

from qiskit.algorithms.gradients import LinCombQFI, LinCombEstimatorGradient

from qiskit.quantum_info import partial_trace

from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import (
    build_ansatz,
    build_init_ansatz_params_vals,
)

from qiskit.algorithms.gibbs_state_preparation.varqite_gibbs_state_builder import (
    VarQiteGibbsStateBuilder,
)
from qiskit.visualization import plot_histogram


def create_ansatz(n: int, reps: int):
    """Create the ansatz.
    Args:
        n: number of qubits
        reps: number of repetitions of the efficient SU2 part in the ansatz
    """
    effsu2 = EfficientSU2(
        N,
        reps=reps,
        insert_barriers=True,
        entanglement="circular",
        parameter_prefix="Ent",
    ).decompose()
    qr = QuantumRegister(N, name="quantum")
    ancilla = QuantumRegister(N, name="a")
    circuit = QuantumCircuit(qr, ancilla)
    Energies = ParameterVector("E", N)

    circuit.barrier()
    for n, energ in enumerate(Energies):
        circuit.ry(energ, qr[n])

    circuit.barrier()
    circuit.cx(qr, ancilla)
    circuit.barrier()

    circuit.append(effsu2, qr)
    circuit.barrier()
    return circuit





def cmaes_warmstart(ansatz, hamiltonian, initial_beta, max_evals):
    """Returns the parameters of the ansatz at a thermal state with temperature `initial_beta`."""
    loss = lambda parameters: free_energy(ansatz, parameters, hamiltonian, initial_beta)
    N = hamiltonian.num_qubits
    initial_point = np.zeros(ansatz.num_parameters)
    initial_point[0:N] = np.ones(N) * np.pi / 2
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


def entropy_fromparams(energies):
    single_entropy = (
        lambda p: -p * np.log(p) - (1 - p) * np.log(1 - p) if p != 0 and p != 1 else 0
    )
    return np.real(sum(single_entropy(np.cos(E / 2) ** 2) for E in energies))


def free_energy(ansatz, parameters, hamiltonian, beta):
    N = hamiltonian.num_qubits
    bound = ansatz.bind_parameters(parameters)
    energy = Statevector(bound).expectation_value(hamiltonian)
    entropy = entropy_fromparams(parameters[:N])
    return np.real(energy - entropy / beta)


def varqite_evolve(hamiltonian, ansatz, init_param_values, steps, delta_beta):
    """Increases the inverse temperature of the ansatz by `delta_beta`."""
    # algorithm_globals.random_seed = 123
    estimator = Estimator()
    qfi = LinCombQFI(estimator)
    gradient = LinCombEstimatorGradient(estimator)
    var_principle = ImaginaryMcLachlanPrinciple(qfi, gradient)
    lse_solver = lambda a, b: np.linalg.lstsq(a, b, rcond=1e-5)[0]
    var_qite = VarQITE(
        ansatz=ansatz,
        initial_parameters=init_param_values,
        variational_principle=var_principle,
        estimator=estimator,
        num_timesteps=steps,
        lse_solver=lse_solver,
    )
    time_problem = TimeEvolutionProblem(
        -hamiltonian ^ ("I" * hamiltonian.num_qubits), delta_beta / 2
    )
    evolved_circuit = var_qite.evolve(time_problem).evolved_state
    return evolved_circuit


def traced_state(circuit):
    N = circuit.num_qubits // 2
    return partial_trace(Statevector(circuit), range(N, 2 * N))


def fidelity(state1, state2):
    return np.real(np.vdot(state1, state2))





if __name__ == "__main__":
    ####Set the parameters
    reps = 2
    beta = 1
    epsilon = 0
    #### Create the ansatz and the Hamiltonian
    hamiltonian = SparsePauliOp.from_list([("ZZ", -1)])
    N = hamiltonian.num_qubits
    ansatz = create_ansatz(N, reps)
    # print(ansatz.decompose().draw())
    #### Warmstart the ansatz at an initial beta
    warmstarted_params, _, fevals = cmaes_warmstart(
        ansatz, hamiltonian, beta * epsilon, 10000
    )
    print(warmstarted_params)
    print(f"Warmstarted parameters with {fevals} function evaluations.")
    print(
        f"After warmstarting we have a max error of {np.abs(traced_state(ansatz.bind_parameters(warmstarted_params),N).data -expected_state(hamiltonian,beta*epsilon)).max()} with the expeted state"
    )
    #### Increase the inverse temperature of the ansatz to the desired one.
    evolved_circuit = varqite_evolve(
        hamiltonian, ansatz, warmstarted_params, 5, beta * (1 - epsilon)
    )
    print("Expected state:")
    printarray(expected_state(hamiltonian, beta))
    print("Actual state:")
    printarray(traced_state(ansatz.bind_parameters(warmstarted_params), N).data)
# # %%
#     print("Parameters of the ansatz:")
#     print(ansatz.bind_parameters(warmstarted_params).draw())
#     print(evolved_circuit.draw())
