from abc import ABC, abstractmethod
import numpy as np
# TODO: make lib package
from .utils import prepend_x0


class GLM(ABC):

    def __init__(
        self, max_iter=100, theta: np.ndarray = np.array([]), alpha=0.01
    ) -> None:
        self.max_iter = max_iter
        self.theta = theta
        self.alpha = alpha

    @abstractmethod
    def h(self, theta: np.ndarray, xi: np.ndarray):
        pass

    def batch_gradient_ascent(self, x: np.ndarray, y: np.ndarray):
        m, n = x.shape

        def next_iteration(theta, x: np.ndarray, y: np.ndarray):
            for j in range(n):
                sum = 0
                for i in range(m):
                    sum += (y[i] - self.h(theta, x[i])) * x[i][j]

                theta[j] += self.alpha * sum

            return theta

        for _ in range(self.max_iter):
            self.theta = next_iteration(self.theta, x, y)

    def stochastic_gradient_ascent(self, x: np.ndarray, y: np.ndarray):
        m, n = x.shape

        for _ in range(self.max_iter):
            for i in range(m):
                for j in range(n):
                    update_rule = (
                        self.alpha * (y[i] - self.h(self.theta, x[i])) * x[i, j]
                    )
                    self.theta[j, 0] += update_rule

    def maximize_likelihood(self, x: np.ndarray, y: np.ndarray):
        self.batch_gradient_ascent(x, y)

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = prepend_x0(x)

        _, n = x.shape
        if self.theta.size == 0:
            self.theta = np.zeros((n, 1))

        self.maximize_likelihood(x, y)

    def predict(self, x):
        x = prepend_x0(x)
        m, _ = x.shape

        y = np.empty(m)

        for i in range(m):
            y[i] = self.h(self.theta, x[i])

        return y
