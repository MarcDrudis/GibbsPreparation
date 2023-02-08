
from __future__ import annotations

from qiskit.algorithms.time_evolvers.variational.variational_principles import VariationalPrinciple
from qiskit.algorithms.gradients import BaseEstimatorGradient, BaseQGT, DerivativeType


def store_info(stored_object:list ,function:callable):
    def wrapped_fun(*args, **kwargs):
        result = function(*args, **kwargs)
        stored_object.append(result)
        return result
    return wrapped_fun

def variationalprinciplestorage(var_prin:VariationalPrinciple):
    
    class VariationalPrincipleStorage(var_prin):
    
        def __init__(
            self,
            qgt: BaseQGT,
            gradient: BaseEstimatorGradient,
        ) -> None:
            """
            Args:
                qgt: Instance of a class used to compute the GQT.
                gradient: Instance of a class used to compute the state gradient.
            """
            super().__init__(qgt,gradient)
            self.stored_qgts = []
            self.stored_gradients = []
            self.metric_tensor = store_info(self.stored_qgts,self.metric_tensor)
            self.evolution_gradient = store_info(self.stored_gradients,self.evolution_gradient)
            
    return VariationalPrincipleStorage