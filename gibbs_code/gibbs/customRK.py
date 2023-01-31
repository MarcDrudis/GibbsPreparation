from scipy.integrate import RK45
import numpy as np


class CustomRK(RK45):
    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        max_step=np.inf,
        rtol=1e-3,
        atol=1e-3,
        vectorized=False,
        first_step=None,
        **extraneous,
    ):

        super().__init__(
            fun,
            t0,
            y0,
            t_bound,
            max_step,
            rtol,
            atol,
            vectorized,
            first_step,
            **extraneous,
        )

    def step(self):
        print(f"Time is:{self.t:.3e}")
        return super().step()
