from __future__ import annotations
from tqdm import trange
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.preparation.varqite import efficientTwoLocalansatz
from gibbs.utils import (
    classical_learn_hamiltonian,
    identity_purification,
    number_of_elements,
    simple_purify_hamiltonian,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.sparse.linalg import expm_multiply


def get_results(folder_path: str) -> list[GibbsResult]:
    """Loads all results in a folder"""
    gibbs_result_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            path = os.path.join(folder_path, file)
            gibbs_result_list.append(GibbsResult.load(path))
    return gibbs_result_list


@dataclass
class GibbsResult:
    """
    Dataclass that stores the result of the VarQITE evolution.
    """

    ansatz_arguments: dict
    parameters: list[np.ndarray]
    coriginal: np.ndarray
    num_qubits: int
    klocality: int
    betas: list[float]
    stored_gradients: list[np.ndarray] = None
    stored_qgts: list[np.ndarray] = None
    shots: int | None = None
    runtime: str | None = None
    cfaulties: list[np.ndarray] | None = None
    periodic: bool = False
    date: str = field(
        default_factory=lambda: datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    )

    def __post_init__(self):
        try:
            if self.cfaulties is None:
                ansatz, _ = efficientTwoLocalansatz(**self.ansatz_arguments)
                self.cfaulties = [
                    classical_learn_hamiltonian(
                        ansatz.bind_parameters(p), self.klocality, self.periodic
                    )
                    for p in self.parameters
                ]
        except:
            pass

    def save(self, path):
        """
        Saves the class as a dictionary into a .npy file.
        """

        np.save(f"{path}_date={self.date}", self.__dict__)

    @classmethod
    def load(self, path):
        """
        Loads a dictionary from a .npy file and returns a VarQITEResult class.
        """
        dictionary = np.load(path, allow_pickle=True).item()
        if "cfaultnorms" in dictionary.keys():
            dictionary.pop("cfaultnorms")
        return self(**dictionary)

    def state_ansatz(self, timestep: int) -> QuantumCircuit:
        return self.ansatz.bind_parameters(self.parameters[timestep])

    def state_vector(self, timestep: int) -> Statevector:
        return Statevector(self.state_ansatz(timestep))

    @property
    def hamiltonian(self):
        """Returns the original Hamiltonian."""
        return self.basis.vector_to_pauli_op(self.coriginal)

    @property
    def ansatz(self):
        """Returns the ansatz."""
        return efficientTwoLocalansatz(**self.ansatz_arguments)[0]

    @property
    def basis(self):
        """Returns the KLocalPauliBasis."""
        return KLocalPauliBasis(self.klocality, self.num_qubits)

    def fidelity(self, timestep: int) -> float:
        """Computes the fidelity between the faulty state at a given timestep with the expected preparation."""
        return state_fidelity(
            self.state_vector(timestep),
            simple_purify_hamiltonian(self.hamiltonian * self.betas[timestep]),
        )

    def local_size(self, k: int, periodic: bool = False):
        return KLocalPauliBasis(k, self.num_qubits, periodic=periodic).size

    def animated_hamiltonian(self, interval: int = 1000, func: callable = np.abs):
        """Creates an animation of the evolution of the Hamiltonian."""
        import matplotlib.pyplot as plt
        from IPython.display import HTML
        from matplotlib.animation import FuncAnimation

        plt.style.use("seaborn-pastel")
        fig = plt.figure()
        scale = 1.1 * max(np.abs(self.coriginal))
        ax = plt.axes(xlim=(0, len(self.cfaulties[0])), ylim=(-scale, scale))

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
            rescaling = 1 / self.betas[i] if self.betas[i] != 0 else 1
            y = func(self.cfaulties[i]) * rescaling
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

    def fidelity_evolution(self, progress_bar: bool = False):
        """Returns the fidelity of the state for each timestep."""
        iterable = trange(len(self.betas)) if progress_bar else range(len(self.betas))
        return np.array([self.fidelity(i) for i in iterable])

    def change_locality(self, new_locality: int):
        new_dict = self.__dict__.copy()
        new_dict.pop("cfaulties")
        new_dict["klocality"] = new_locality
        if new_locality > self.klocality:
            new_dict["coriginal"] = np.zeros(
                self.local_size(new_locality), dtype=np.complex128
            )
            new_dict["coriginal"][: self.coriginal.size] = self.coriginal
        else:
            new_dict["coriginal"] = self.coriginal[: self.local_size(new_locality)]
        return GibbsResult(**new_dict)
