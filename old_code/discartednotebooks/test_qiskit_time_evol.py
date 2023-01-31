{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from gibbs.utils import create_hamiltonian_lattice, create_heisenberg\n",
    "from gibbs.preparation.varqite import efficientTwoLocalansatz\n",
    "from qiskit.opflow import PauliSumOp\n",
    "from qiskit.quantum_info import SparsePauliOp\n",
    "from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis\n",
    "from gibbs.dataclass import GibbsResult"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_qubits = 8\n",
    "horiginal = create_heisenberg(num_qubits,1/4,-1)\n",
    "\n",
    "coriginal = KLocalPauliBasis(2,num_qubits).pauli_to_vector(horiginal)\n",
    "ansatz_arguments = {\"num_qubits\":num_qubits,\"depth\":2,\"entanglement\":\"reverse_linear\",\"su2_gates\":[\"ry\"],\"ent_gates\":[\"cx\"]}\n",
    "ansatz,x0 = efficientTwoLocalansatz(**ansatz_arguments)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "varqite = VarQITE(ansatz,x0,backend=\"TURBO\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.10.7 ('JulienVarQITE')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.7"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "e0c64f3bcdec0072c9c7dd86851c5f64bc0ac47e1b8f552608c742c012e3e497"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
