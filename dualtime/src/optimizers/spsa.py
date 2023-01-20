import logging
import numpy as np

logger = logging.getLogger(__name__)


class SPSA:
    def __init__(
        self,
        maxiter,
        learning_rate,
        perturbation,
        batch_size=1,
        losstol=1e-5,
        gradtol=1e-3,
        averaging=1,
    ):
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.batch_size = batch_size
        # self.averaging = averaging
        self.losstol = losstol
        self.gradtol = gradtol

    def __call__(self, loss_and_gradient, x0):

        if not hasattr(self.learning_rate, "__iter__"):
            maxiter = self.maxiter
            learning_rate = self.learning_rate
            eta = iter(np.ones(maxiter) * learning_rate)
        else:
            eta = iter(self.learning_rate)

        x = x0
        xs = [x0]
        losses = []
        gradients = []

        print(x0)
        fun = lambda x: loss_and_gradient(x, return_gradient=False)
        eps = 0.01

        # for _ in range(2):
        for _ in range(self.maxiter):
            # deltas = [(-1) ** np.random.randint(0, 2, size=x0.size) for _ in range(self.batch_size)]
            # print(deltas)
            # plus = [x + self.perturbation * delta for delta in deltas]
            # minus = [x - self.perturbation * delta for delta in deltas]

            # results = loss_and_gradient(plus + minus, return_gradient=False)
            # print(results[0], results[self.batch_size])

            # grads = [
            #     (results[i] - results[self.batch_size + i]) / (2 * self.perturbation) * delta
            #     for i, delta in enumerate(deltas)
            # ]

            # gradient = np.mean(grads, axis=0)
            # print(gradient)
            delta = (-1) ** np.random.randint(0, 2, x0.size)
            gradient = (fun(x + eps * delta) - fun(x - eps * delta)) / (2 * eps) * delta
            # x = x - eta * grad
            # update iterate
            x = x - next(eta) * gradient

            losses.append(loss_and_gradient(x, return_gradient=False))
            gradients.append(np.linalg.norm(gradient))
            xs.append(x)

            # logger.info("-- Loss: %f, gradientnorm: %f", loss, np.linalg.norm(gradient))

            # if len(losses) >= 2 and np.abs(losses[-1] - losses[-2]) < self.losstol:
            #     print("-- Reached losstol.")
            #     break

            # if np.linalg.norm(gradient) < self.gradtol:
            #     print("-- Reached tolerance.")
            #     break

        return x, xs, losses, gradients
