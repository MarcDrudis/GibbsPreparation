from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
from gibbs.preparation.varqite import efficientTwoLocalansatz
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.utils import number_of_elements, identity_purification,classical_learn_hamiltonian
from qiskit.quantum_info import state_fidelity, Statevector
from scipy.sparse.linalg import expm_multiply


@dataclass
class GibbsResult:
    """
    Dataclass that stores the result of theVarQITE evolution.
    """

    ansatz_arguments: dict
    parameters: list[np.ndarray]
    coriginal: np.ndarray
    num_qubits: int
    klocality: int
    betas: list[float]
    cfaulties: list[np.ndarray] | None = None
    date: str = field(
        default_factory=lambda: datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    )

    def __post_init__(self):
        if self.cfaulties is None:
            ansatz, _ = efficientTwoLocalansatz(**self.ansatz_arguments)
            self.cfaulties = [classical_learn_hamiltonian(ansatz.bind_parameters(p), self.klocality) for p in self.parameters]

    def save(self, path):
        """
        Saves the class as a dictionary into a .npy file.
        """

        np.save(f"{path}_date={self.date}", self.__dict__)

    @classmethod
    def load(cls, path):
        """
        Loads a dictionary from a .npy file and returns a VarQITEResult class.
        """
        dictionary = np.load(path, allow_pickle=True).item()
        if "cfaultnorms" in dictionary.keys():
            dictionary.pop("cfaultnorms")            
        return cls(**dictionary)

    def animated_hamiltonian(self, interval: int = 1000, func: callable = np.abs):
        """Creates an animation of the evolution of the Hamiltonian."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from IPython.display import HTML

        plt.style.use("seaborn-pastel")
        fig = plt.figure()
        ax = plt.axes(xlim=(0, len(self.cfaulties[0])), ylim=(-1, 1))

        # add another axes at the top left corner of the figure
        axtext = fig.add_axes([0.0, 0.95, 0.1, 0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis("off")
        time = axtext.text(0.5, 0.5, "beta=" + str(0), ha="left", va="top")

        ax.stairs(
            values=func(self.coriginal),
            edges=np.arange(len(self.coriginal) + 1) - 0.5,
            lw=3,
        )
        for i in range(1, self.klocality):
            ax.axvline(
                number_of_elements(i, self.num_qubits) - 0.5,
                color="gray",
                lw=1,
                linestyle="dashed",
            )

        (line,) = ax.plot([], [], marker="o", linestyle="None", markersize=5)
        ax.set_xlabel("Pauli Terms")
        ax.set_ylabel("Coefficients")
        fig.suptitle("Weight of Pauli terms in Faulty Hamiltonian")

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i):
            x = range(len(self.cfaulties[-1]))
            y = func(self.cfaulties[i])
            line.set_data(x, y)
            time.set_text(f"beta={self.betas[i]:.3f}")
            return (
                line,
                time,
            )

        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(self.cfaulties),
            interval=interval,
            blit=True,
        )
        plt.close(fig)
        return HTML(anim.to_html5_video())

    def fidelity_evolution(self):
        """Returns the fidelity of the state for each timestep."""

        fidelities = [None] * len(self.betas)
        hamiltonian = KLocalPauliBasis(self.klocality,self.num_qubits).vector_to_pauli_op(self.coriginal)^("I"*self.num_qubits)
        expected_states = expm_multiply(
            -hamiltonian.to_matrix(sparse=True),
            identity_purification(self.num_qubits).data,
            start=0,
            stop=self.betas[-1] / 2,
            num=len(self.betas),
            endpoint=True
        )
        
        for i, p in enumerate(self.parameters):
            faulty_state = Statevector(
                efficientTwoLocalansatz(**self.ansatz_arguments)[0].bind_parameters(p)
            )
            expected_state = Statevector(expected_states[i])/np.linalg.norm(expected_states[i])
            fidelities[i] = state_fidelity(faulty_state, expected_state)
        return fidelities
