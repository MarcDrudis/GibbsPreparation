import logging
import numpy as np

logger = logging.getLogger(__name__)


class GradientDescent:
    def __init__(
        self, maxiter, learning_rate, blocking=False, losstol=1e-5, gradtol=1e-3, momentum=0
    ):
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.blocking = blocking
        self.losstol = losstol
        self.gradtol = gradtol
        self.momentum = momentum

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

        for i in range(self.maxiter):
            loss, gradient = loss_and_gradient(x)

            # apply momentum
            if self.momentum > 0 and i > 0:
                moment = self.momentum * gradients[-1] + (1 - self.momentum) * gradient
            else:
                moment = gradient

            # update iterate
            x = x - next(eta) * moment

            # blocking criterion
            if self.blocking and len(losses) > 1:
                if loss - losses[-1] > 0.01:
                    logger.info("-- Rejecting GD step.")
                    continue

            losses.append(loss)
            gradients.append(np.linalg.norm(gradient))
            xs.append(x)

            logger.info("-- Loss: %f, gradientnorm: %f", loss, np.linalg.norm(gradient))

            if len(losses) >= 2 and np.abs(losses[-1] - losses[-2]) < self.losstol:
                print("-- Reached losstol.")
                break

            if np.linalg.norm(gradient) < self.gradtol:
                print("-- Reached tolerance.")
                break

        return x, xs, losses, gradients
